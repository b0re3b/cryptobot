import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from data.db import DatabaseManager
from utils.config import *
class CryptoCycles:

    def __init__(self):

        self.db_connection = DatabaseManager()

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

        self.cached_processed_data = {}

    def load_processed_data(self, symbol: str, timeframe: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> dict[Any, Any] | None | Any:

        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"

        if cache_key in self.cached_processed_data:
            return self.cached_processed_data[cache_key]

        # Load pre-processed data from storage manager
        processed_data = self.db_connection.get_btc_arima_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        # Cache for future use
        self.cached_processed_data[cache_key] = processed_data

        return processed_data

    def calculate_token_specific_cycle_features(self,
                                                processed_data: pd.DataFrame,
                                                symbol: str) -> pd.DataFrame:

        symbol = symbol.upper().replace('USDT', '').replace('USD', '')

        if symbol == 'BTC':
            return self.calculate_btc_halving_cycle_features(processed_data)
        elif symbol == 'ETH':
            return self.calculate_eth_event_cycle_features(processed_data)
        elif symbol == 'SOL':
            return self.calculate_sol_event_cycle_features(processed_data)
        else:
            # For other tokens, return the original data without specific features
            print(f"Warning: No specific cycle features available for {symbol}. Returning original data.")
            return processed_data

    def create_cyclical_features(self, processed_data: pd.DataFrame, symbol: str) -> pd.DataFrame:

        # First, add general cyclical features (common to all cryptocurrencies)
        result_df = processed_data.copy()

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

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

        # Add market phase features
        market_phase_df = self.detect_market_phase(result_df)
        # Get only the market_phase column
        if 'market_phase' in market_phase_df.columns:
            result_df['market_phase'] = market_phase_df['market_phase']
            # One-hot encode the market phase
            phase_dummies = pd.get_dummies(result_df['market_phase'], prefix='phase')
            result_df = pd.concat([result_df, phase_dummies], axis=1)

        # Identify bull/bear cycles
        try:
            bull_bear_df = self.identify_bull_bear_cycles(result_df)
            # Add cycle state and ID
            if 'cycle_state' in bull_bear_df.columns:
                result_df['cycle_state'] = bull_bear_df['cycle_state']
                result_df['cycle_id'] = bull_bear_df['cycle_id']
                # One-hot encode the cycle state
                state_dummies = pd.get_dummies(result_df['cycle_state'], prefix='state')
                result_df = pd.concat([result_df, state_dummies], axis=1)
        except Exception as e:
            # If bull/bear cycle detection fails, log the error and continue
            print(f"Error detecting bull/bear cycles: {e}")

        # Then add token-specific cycle features
        result_df = self.calculate_token_specific_cycle_features(result_df, symbol)

        # Try to find and add optimal cycle features
        try:
            optimal_cycle_length, cycle_strength = self.find_optimal_cycle_length(result_df)
            # If a strong cycle is found, add related features
            if cycle_strength > 0.3:  # Only add if moderately strong cycle detected
                result_df['optimal_cycle_length'] = optimal_cycle_length
                result_df['optimal_cycle_strength'] = cycle_strength
                # Calculate days into the optimal cycle
                result_df['days_into_optimal_cycle'] = result_df.index.dayofyear % optimal_cycle_length
                # Normalized position in the optimal cycle (0 to 1)
                result_df['optimal_cycle_phase'] = result_df['days_into_optimal_cycle'] / optimal_cycle_length
                # Create sine and cosine features for the optimal cycle
                cycle_phase = result_df['optimal_cycle_phase'] * 2 * np.pi
                result_df['optimal_cycle_sin'] = np.sin(cycle_phase)
                result_df['optimal_cycle_cos'] = np.cos(cycle_phase)
        except Exception as e:
            # If optimal cycle detection fails, log the error and continue
            print(f"Error finding optimal cycle: {e}")

        # Drop the original categorical columns if one-hot encoding was successful
        if 'phase_accumulation' in result_df.columns:
            result_df = result_df.drop(columns=['market_phase'], errors='ignore')
        if 'state_bull' in result_df.columns:
            result_df = result_df.drop(columns=['cycle_state'], errors='ignore')

        return result_df

    def find_optimal_cycle_length(self, processed_data: pd.DataFrame,
                                  min_period: int = 7,
                                  max_period: int = 365) -> Tuple[int, float]:

        # Ensure we have the required columns
        if 'close' not in processed_data.columns:
            raise ValueError("Required column 'close' not found in processed_data")

        # Ensure datetime index and resample to daily frequency if needed
        df = processed_data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Create a daily price series if data is not already daily
        if df.index.to_series().diff().mean().total_seconds() != 86400:
            daily_df = df.resample('D').last()
            price_series = daily_df['close'].dropna()
        else:
            price_series = df['close'].dropna()

        # If data is limited, adjust the maximum period
        if len(price_series) < max_period * 2:
            max_period = min(len(price_series) // 2, max_period)

        # Calculate price returns for better stationarity
        returns = price_series.pct_change().dropna()

        # Detect seasonality using autocorrelation
        correlations = []

        # Calculate autocorrelation for different lags
        for lag in range(min_period, max_period + 1):
            if len(returns) > lag:
                # Calculate autocorrelation
                corr = returns.autocorr(lag=lag)
                correlations.append((lag, corr))

        # If no valid correlations were found, return default values
        if not correlations:
            return (max_period, 0.0)

        # Convert to DataFrame for easier manipulation
        corr_df = pd.DataFrame(correlations, columns=['lag', 'autocorrelation'])

        # Find local maxima in the autocorrelation function
        local_maxima = []
        for i in range(1, len(corr_df) - 1):
            if (corr_df['autocorrelation'].iloc[i] > corr_df['autocorrelation'].iloc[i - 1] and
                    corr_df['autocorrelation'].iloc[i] > corr_df['autocorrelation'].iloc[i + 1]):
                local_maxima.append((
                    corr_df['lag'].iloc[i],
                    corr_df['autocorrelation'].iloc[i]
                ))

        # If no local maxima found, return the highest correlation
        if not local_maxima:
            best_lag = corr_df.loc[corr_df['autocorrelation'].idxmax()]
            return (int(best_lag['lag']), float(best_lag['autocorrelation']))

        # Sort local maxima by correlation strength
        local_maxima.sort(key=lambda x: x[1], reverse=True)

        # Return the highest correlation local maximum
        return (int(local_maxima[0][0]), float(local_maxima[0][1]))

    def calculate_cycle_roi(self, processed_data: pd.DataFrame,
                            symbol: str,
                            cycle_type: str = 'auto',
                            normalized: bool = True) -> pd.DataFrame:

        # Create a copy of the input DataFrame
        df = processed_data.copy()

        # Ensure the DataFrame has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Ensure we have the required columns
        if 'close' not in df.columns:
            raise ValueError("Required column 'close' not found in processed_data")

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

        # Initialize result DataFrame
        result_df = pd.DataFrame(index=df.index)
        result_df['close'] = df['close']
        result_df['date'] = df.index

        # Calculate ROI based on cycle type
        if cycle_type == 'halving':
            # For BTC halving cycles
            halving_df = self.calculate_btc_halving_cycle_features(df)

            # Group by cycle number
            for cycle_num in halving_df['cycle_number'].unique():
                if pd.isna(cycle_num) or cycle_num == 0:
                    continue

                cycle_data = halving_df[halving_df['cycle_number'] == cycle_num]
                if len(cycle_data) > 0:
                    cycle_start_price = cycle_data['close'].iloc[0]
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

        elif cycle_type == 'network_upgrade':
            # For ETH network upgrades
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
                    event_start_price = event_data['close'].iloc[0]
                    event_start_date = event_data.index[0]

                    # Calculate ROI for this event cycle
                    result_df.loc[event_data.index, f'event_{event_name}_roi'] = (
                            event_data['close'] / event_start_price - 1
                    )
                    result_df.loc[event_data.index, f'event_{event_name}_days'] = (
                        (event_data.index - event_start_date).days
                    )

        elif cycle_type == 'ecosystem_event':
            # For SOL ecosystem events
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
                    event_start_price = event_data['close'].iloc[0]
                    event_start_date = event_data.index[0]

                    # Calculate ROI for this event cycle
                    result_df.loc[event_data.index, f'event_{event_name}_roi'] = (
                            event_data['close'] / event_start_price - 1
                    )
                    result_df.loc[event_data.index, f'event_{event_name}_days'] = (
                        (event_data.index - event_start_date).days
                    )

        elif cycle_type == 'bull_bear':
            # For general bull/bear cycles
            try:
                bull_bear_df = self.identify_bull_bear_cycles(df)

                # Process cycle summary if available
                if hasattr(bull_bear_df, 'cycles_summary') and len(bull_bear_df.cycles_summary) > 0:
                    cycles_summary = bull_bear_df.cycles_summary

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
                            cycle_start_price = cycle_data['close'].iloc[0]

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
                                        (cycle_data.index - start_date).days / cycle_duration
                                )

                # Add current cycle information
                current_cycle_mask = bull_bear_df['cycle_id'] == bull_bear_df['cycle_id'].max()
                if current_cycle_mask.any():
                    current_cycle = bull_bear_df[current_cycle_mask]
                    current_cycle_state = current_cycle['cycle_state'].iloc[0]
                    current_cycle_id = current_cycle['cycle_id'].iloc[0]
                    current_cycle_start_price = current_cycle['cycle_start_price'].iloc[0]
                    current_cycle_max_price = current_cycle['cycle_max_price'].max()
                    current_cycle_min_price = current_cycle['cycle_min_price'].min()

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

            except Exception as e:
                print(f"Error in calculating bull/bear cycle ROI: {e}")
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

                # Calculate ROI between each pair of turning points
                for i in range(len(turning_points) - 1):
                    start_date = turning_points[i]
                    end_date = turning_points[i + 1]

                    period_mask = (df.index >= start_date) & (df.index <= end_date)
                    period_data = df.loc[period_mask]

                    if len(period_data) > 0:
                        period_start_price = period_data['close'].iloc[0]
                        point_type = 'peak_to_trough' if start_date in peaks else 'trough_to_peak'

                        result_df.loc[period_data.index, f'turning_point_{i}_{point_type}_roi'] = (
                                period_data['close'] / period_start_price - 1
                        )

        elif cycle_type == 'custom':
            # For custom defined cycles
            # Default to optimal cycles detected by the find_optimal_cycle_length method
            try:
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
                            avg_returns_by_cycle_day[day] = day_returns.mean()

                    # Apply the average returns for each day of the cycle
                    for day, avg_return in avg_returns_by_cycle_day.items():
                        result_df.loc[result_df['day_of_cycle'] == day, 'cycle_day_avg_return'] = avg_return

                    # Calculate cumulative expected return based on cycle day
                    result_df['cycle_expected_return'] = result_df['cycle_day_avg_return'].cumsum()

                    # Normalize the cycle phase (0 to 1)
                    result_df['cycle_phase'] = result_df['day_of_cycle'] / optimal_cycle_length

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
                            result_df.loc[phase_data.index, f'{phase}_avg_return'] = phase_returns.mean()
                            result_df.loc[phase_data.index, f'{phase}_volatility'] = phase_returns.std()

            except Exception as e:
                print(f"Error in calculating custom cycle ROI: {e}")

        # Normalize ROI values if requested
        if normalized:
            # Identify ROI columns
            roi_columns = [col for col in result_df.columns if 'roi' in col.lower()]

            # Calculate z-score normalization for each ROI column
            for col in roi_columns:
                mean_val = result_df[col].mean()
                std_val = result_df[col].std()

                if std_val > 0:  # Avoid division by zero
                    result_df[f'{col}_normalized'] = (result_df[col] - mean_val) / std_val

        return result_df



    def detect_cycle_anomalies(self, processed_data: pd.DataFrame,
                               symbol: str,
                               cycle_type: str = 'auto') -> pd.DataFrame:

        # Create a copy of the input DataFrame
        result_df = processed_data.copy()

        # Ensure we have the required columns
        if 'close' not in result_df.columns:
            raise ValueError("Required column 'close' not found in processed_data")

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Clean symbol format
        clean_symbol = symbol.upper().replace('USDT', '').replace('USD', '')

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

        # Initialize the anomalies DataFrame
        anomalies = pd.DataFrame(index=result_df.index)
        anomalies['date'] = anomalies.index
        anomalies['symbol'] = symbol
        anomalies['anomaly_detected'] = False
        anomalies['anomaly_type'] = None
        anomalies['significance_score'] = 0.0
        anomalies['description'] = None

        # Calculate baseline metrics for anomaly detection
        result_df['price_change_1d'] = result_df['close'].pct_change(1)
        result_df['price_change_7d'] = result_df['close'].pct_change(7)
        result_df['price_change_30d'] = result_df['close'].pct_change(30)
        result_df['volatility_14d'] = result_df['close'].pct_change().rolling(window=14).std()

        # Calculate rolling metrics for baseline comparison
        result_df['price_change_1d_zscore'] = (
                (result_df['price_change_1d'] - result_df['price_change_1d'].rolling(window=365).mean()) /
                result_df['price_change_1d'].rolling(window=365).std()
        )
        result_df['price_change_7d_zscore'] = (
                (result_df['price_change_7d'] - result_df['price_change_7d'].rolling(window=365).mean()) /
                result_df['price_change_7d'].rolling(window=365).std()
        )
        result_df['volatility_14d_zscore'] = (
                (result_df['volatility_14d'] - result_df['volatility_14d'].rolling(window=365).mean()) /
                result_df['volatility_14d'].rolling(window=365).std()
        )

        # Add cycle-specific features based on cycle_type
        if cycle_type == 'halving':
            # Get halving cycle features
            halving_df = self.calculate_btc_halving_cycle_features(result_df)

            # Merge the relevant columns with result_df
            for col in ['halving_cycle_phase', 'days_since_last_halving', 'days_to_next_halving', 'cycle_number']:
                if col in halving_df.columns:
                    result_df[col] = halving_df[col]

            # Group historical data by cycle number for comparison
            if 'cycle_number' in result_df.columns and 'halving_cycle_phase' in result_df.columns:
                cycles_data = {}
                current_cycle = result_df['cycle_number'].max()

                for cycle in result_df['cycle_number'].unique():
                    if cycle < current_cycle:  # Historical cycles
                        cycle_data = result_df[result_df['cycle_number'] == cycle]
                        cycles_data[cycle] = cycle_data

                # Get current cycle data
                current_cycle_data = result_df[result_df['cycle_number'] == current_cycle]

                # For each point in the current cycle, compare with historical cycles at the same phase
                for idx, row in current_cycle_data.iterrows():
                    if pd.isna(row['halving_cycle_phase']):
                        continue

                    current_phase = row['halving_cycle_phase']
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
                                historical_values.extend(similar_phase_data['price_change_since_halving'].tolist())

                    # If we have enough historical data points for comparison
                    if len(historical_values) >= 3:
                        historical_mean = np.mean(historical_values)
                        historical_std = np.std(historical_values)

                        # Calculate z-score compared to historical cycles
                        if 'price_change_since_halving' in current_cycle_data.columns and historical_std > 0:
                            current_value = row.get('price_change_since_halving', 0)
                            z_score = (current_value - historical_mean) / historical_std

                            # Check for significant deviation
                            if abs(z_score) > 2.0:  # More than 2 standard deviations
                                anomalies.loc[idx, 'anomaly_detected'] = True
                                anomalies.loc[idx, 'significance_score'] = abs(z_score)

                                if z_score > 0:
                                    anomalies.loc[idx, 'anomaly_type'] = 'higher_than_historical'
                                    anomalies.loc[idx, 'description'] = (
                                        f"Price is {z_score:.2f} std devs higher than historical cycles "
                                        f"at similar phase ({current_phase:.2f})"
                                    )
                                else:
                                    anomalies.loc[idx, 'anomaly_type'] = 'lower_than_historical'
                                    anomalies.loc[idx, 'description'] = (
                                        f"Price is {abs(z_score):.2f} std devs lower than historical cycles "
                                        f"at similar phase ({current_phase:.2f})"
                                    )

        elif cycle_type == 'bull_bear':
            # Get bull/bear cycle information
            bull_bear_df = self.identify_bull_bear_cycles(result_df)

            # Merge the relevant columns with result_df
            for col in ['cycle_state', 'cycle_id', 'days_in_cycle', 'cycle_max_roi', 'cycle_max_drawdown']:
                if col in bull_bear_df.columns:
                    result_df[col] = bull_bear_df[col]

            # Check for anomalies in ROI or drawdown compared to typical bull/bear cycles
            if 'cycle_state' in result_df.columns and 'cycle_id' in result_df.columns:
                # Group by cycle_state to get typical metrics for bull and bear markets
                if hasattr(bull_bear_df, 'cycles_summary') and not bull_bear_df.cycles_summary.empty:
                    cycles_summary = bull_bear_df.cycles_summary

                    # Get metrics by cycle type
                    bull_cycles = cycles_summary[cycles_summary['cycle_type'] == 'bull']
                    bear_cycles = cycles_summary[cycles_summary['cycle_type'] == 'bear']

                    # Calculate average duration and ROI metrics
                    if not bull_cycles.empty:
                        avg_bull_duration = bull_cycles['duration_days'].mean()
                        avg_bull_roi = bull_cycles['max_roi'].mean()
                        std_bull_roi = bull_cycles['max_roi'].std()

                    if not bear_cycles.empty:
                        avg_bear_duration = bear_cycles['duration_days'].mean()
                        avg_bear_drawdown = bear_cycles['max_drawdown'].mean()
                        std_bear_drawdown = bear_cycles['max_drawdown'].std()

                    # Identify current cycle
                    current_cycle_id = result_df['cycle_id'].max()
                    current_cycle_data = result_df[result_df['cycle_id'] == current_cycle_id]

                    if not current_cycle_data.empty:
                        current_state = current_cycle_data['cycle_state'].iloc[0]
                        current_duration = current_cycle_data['days_in_cycle'].max()

                        # Check for duration anomalies
                        if current_state == 'bull' and 'avg_bull_duration' in locals():
                            if current_duration > 1.5 * avg_bull_duration:
                                # Extended bull market
                                for idx in current_cycle_data.index[-30:]:  # Last 30 days
                                    anomalies.loc[idx, 'anomaly_detected'] = True
                                    anomalies.loc[idx, 'anomaly_type'] = 'extended_bull_market'
                                    anomalies.loc[idx, 'significance_score'] = current_duration / avg_bull_duration
                                    anomalies.loc[idx, 'description'] = (
                                        f"Extended bull market: {current_duration} days vs. "
                                        f"typical {avg_bull_duration:.0f} days"
                                    )

                        elif current_state == 'bear' and 'avg_bear_duration' in locals():
                            if current_duration > 1.5 * avg_bear_duration:
                                # Extended bear market
                                for idx in current_cycle_data.index[-30:]:  # Last 30 days
                                    anomalies.loc[idx, 'anomaly_detected'] = True
                                    anomalies.loc[idx, 'anomaly_type'] = 'extended_bear_market'
                                    anomalies.loc[idx, 'significance_score'] = current_duration / avg_bear_duration
                                    anomalies.loc[idx, 'description'] = (
                                        f"Extended bear market: {current_duration} days vs. "
                                        f"typical {avg_bear_duration:.0f} days"
                                    )

                        # Check for ROI/drawdown anomalies
                        if current_state == 'bull' and 'std_bull_roi' in locals() and std_bull_roi > 0:
                            current_roi = current_cycle_data['cycle_max_roi'].max()
                            roi_z_score = (current_roi - avg_bull_roi) / std_bull_roi

                            if abs(roi_z_score) > 2.0:
                                anomaly_type = 'stronger_bull' if roi_z_score > 0 else 'weaker_bull'
                                for idx in current_cycle_data.index[-15:]:  # Last 15 days
                                    anomalies.loc[idx, 'anomaly_detected'] = True
                                    anomalies.loc[idx, 'anomaly_type'] = anomaly_type
                                    anomalies.loc[idx, 'significance_score'] = abs(roi_z_score)
                                    anomalies.loc[idx, 'description'] = (
                                        f"{'Stronger' if roi_z_score > 0 else 'Weaker'} than typical bull market: "
                                        f"{current_roi:.1%} ROI vs. typical {avg_bull_roi:.1%}"
                                    )

                        elif current_state == 'bear' and 'std_bear_drawdown' in locals() and std_bear_drawdown > 0:
                            current_drawdown = current_cycle_data['cycle_max_drawdown'].min()
                            drawdown_z_score = (current_drawdown - avg_bear_drawdown) / std_bear_drawdown

                            if abs(drawdown_z_score) > 2.0:
                                anomaly_type = 'milder_bear' if drawdown_z_score > 0 else 'severe_bear'
                                for idx in current_cycle_data.index[-15:]:  # Last 15 days
                                    anomalies.loc[idx, 'anomaly_detected'] = True
                                    anomalies.loc[idx, 'anomaly_type'] = anomaly_type
                                    anomalies.loc[idx, 'significance_score'] = abs(drawdown_z_score)
                                    anomalies.loc[idx, 'description'] = (
                                        f"{'Milder' if drawdown_z_score > 0 else 'More severe'} than typical bear market: "
                                        f"{current_drawdown:.1%} drawdown vs. typical {avg_bear_drawdown:.1%}"
                                    )

        elif cycle_type == 'network_upgrade' or cycle_type == 'ecosystem_event':
            # For ETH or SOL, detect anomalies around significant events
            events = self.get_significant_events_for_symbol(clean_symbol)

            # If we have events data
            if events:
                # Analyze volatility and price changes around events
                for event in events:
                    # Skip if event data is not a dictionary (for BTC it's just dates)
                    if not isinstance(event, dict):
                        continue

                    event_date = pd.Timestamp(event['date'])
                    event_name = event['name']

                    # Skip events that are too recent or future events
                    if event_date > result_df.index.max():
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
                        continue

                    # Calculate metrics
                    pre_event_volatility = pre_event_data[
                        'volatility_14d'].mean() if 'volatility_14d' in pre_event_data.columns else 0
                    post_event_volatility = post_event_data[
                        'volatility_14d'].mean() if 'volatility_14d' in post_event_data.columns else 0

                    # Price change from event day
                    if event_date in result_df.index:
                        event_price = result_df.loc[event_date, 'close']
                        post_event_data['price_change_from_event'] = post_event_data['close'] / event_price - 1

                        # Check for significant price changes after the event
                        for idx, row in post_event_data.iterrows():
                            days_after_event = (idx - event_date).days

                            # If price change is >10% within 7 days or >20% within 30 days of the event
                            if (days_after_event <= 7 and abs(row['price_change_from_event']) > 0.1) or \
                                    (days_after_event > 7 and abs(row['price_change_from_event']) > 0.2):
                                anomalies.loc[idx, 'anomaly_detected'] = True
                                anomalies.loc[idx, 'anomaly_type'] = 'significant_post_event_move'
                                anomalies.loc[idx, 'significance_score'] = abs(
                                    row['price_change_from_event']) * 5  # Scale for comparison
                                anomalies.loc[idx, 'description'] = (
                                    f"Significant price change of {row['price_change_from_event']:.1%} "
                                    f"{days_after_event} days after {event_name}"
                                )

                    # Check for volatility anomalies
                    if post_event_volatility > 1.5 * pre_event_volatility:
                        # Increased volatility after event
                        for idx in post_event_data.index[:10]:  # First 10 days after event
                            anomalies.loc[idx, 'anomaly_detected'] = True
                            anomalies.loc[idx, 'anomaly_type'] = 'increased_post_event_volatility'
                            anomalies.loc[idx, 'significance_score'] = post_event_volatility / pre_event_volatility
                            anomalies.loc[idx, 'description'] = (
                                f"Increased volatility after {event_name}: "
                                f"{post_event_volatility:.4f} vs pre-event {pre_event_volatility:.4f}"
                            )

        # General price anomalies (applicable to all cycle types)
        for idx, row in result_df.iterrows():
            # Skip rows with insufficient data for z-scores
            if pd.isna(row.get('price_change_1d_zscore')) or pd.isna(row.get('volatility_14d_zscore')):
                continue

            # 1. Check for extreme daily price changes
            if abs(row['price_change_1d_zscore']) > 3.0:  # More than 3 standard deviations
                anomalies.loc[idx, 'anomaly_detected'] = True
                anomalies.loc[idx, 'anomaly_type'] = 'extreme_daily_move'
                anomalies.loc[idx, 'significance_score'] = abs(row['price_change_1d_zscore'])
                direction = "up" if row['price_change_1d'] > 0 else "down"
                anomalies.loc[idx, 'description'] = (
                    f"Extreme daily price move {direction} ({row['price_change_1d']:.1%}), "
                    f"z-score: {row['price_change_1d_zscore']:.2f}"
                )

            # 2. Check for extreme volatility
            if row['volatility_14d_zscore'] > 3.0:  # More than 3 standard deviations
                # Only mark as anomaly if not already detected for price change
                if not anomalies.loc[idx, 'anomaly_detected']:
                    anomalies.loc[idx, 'anomaly_detected'] = True
                    anomalies.loc[idx, 'anomaly_type'] = 'extreme_volatility'
                    anomalies.loc[idx, 'significance_score'] = row['volatility_14d_zscore']
                    anomalies.loc[idx, 'description'] = (
                        f"Extreme volatility detected ({row['volatility_14d']:.4f}), "
                        f"z-score: {row['volatility_14d_zscore']:.2f}"
                    )

            # 3. Check for volatility collapse (can precede large moves)
            if row['volatility_14d_zscore'] < -2.0:  # More than 2 standard deviations below mean
                # Only mark as anomaly if not already detected
                if not anomalies.loc[idx, 'anomaly_detected']:
                    anomalies.loc[idx, 'anomaly_detected'] = True
                    anomalies.loc[idx, 'anomaly_type'] = 'volatility_collapse'
                    anomalies.loc[idx, 'significance_score'] = abs(row['volatility_14d_zscore'])
                    anomalies.loc[idx, 'description'] = (
                        f"Unusually low volatility ({row['volatility_14d']:.4f}), "
                        f"z-score: {row['volatility_14d_zscore']:.2f}"
                    )

        # Filter to only include detected anomalies
        anomalies = anomalies[anomalies['anomaly_detected']]

        # Sort by significance score
        anomalies = anomalies.sort_values('significance_score', ascending=False)

        return anomalies

    def get_significant_events_for_symbol(self, symbol: str) -> List:

        symbol = symbol.upper().replace('USDT', '').replace('USD', '')
        return self.symbol_events_map.get(symbol, [])

    def compare_current_to_historical_cycles(self, processed_data: pd.DataFrame,
                                             symbol: str,
                                             cycle_type: str = 'auto',
                                             normalize: bool = True) -> Dict:

        symbol = symbol.upper().replace('USDT', '').replace('USD', '')

        # Determine the appropriate cycle type if 'auto' is specified
        if cycle_type == 'auto':
            if symbol == 'BTC':
                cycle_type = 'halving'
            elif symbol == 'ETH':
                cycle_type = 'network_upgrade'
            elif symbol == 'SOL':
                cycle_type = 'ecosystem_event'
            else:
                cycle_type = 'bull_bear'

        # Ensure the DataFrame has a datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Identify cycles in the data based on cycle_type
        if cycle_type == 'bull_bear':
            # Use the bull_bear cycle identification logic
            cycles_data = self.identify_bull_bear_cycles(processed_data)
            cycle_column = 'cycle_id'
        elif cycle_type == 'halving' and symbol == 'BTC':
            # Use halving cycles for BTC
            cycles_data = self.calculate_btc_halving_cycle_features(processed_data)
            cycle_column = 'cycle_number'
        elif cycle_type == 'network_upgrade' and symbol == 'ETH':
            # For ETH, use network upgrades as cycle boundaries
            # Implementation would depend on how ETH cycles are defined
            cycles_data = processed_data.copy()
            # This is just a placeholder - the actual implementation would need
            # to process ETH network upgrade cycles
            cycle_column = 'cycle_id'
        elif cycle_type == 'ecosystem_event' and symbol == 'SOL':
            # For SOL, use ecosystem events as cycle boundaries
            # Implementation would depend on how SOL cycles are defined
            cycles_data = processed_data.copy()
            # This is just a placeholder - the actual implementation would need
            # to process SOL ecosystem event cycles
            cycle_column = 'cycle_id'
        else:
            # Default to bull/bear cycles for unknown combinations
            cycles_data = self.identify_bull_bear_cycles(processed_data)
            cycle_column = 'cycle_id'

        # Extract the current cycle
        if cycle_column in cycles_data.columns:
            current_cycle_id = cycles_data[cycle_column].iloc[-1]
            current_cycle_data = cycles_data[cycles_data[cycle_column] == current_cycle_id]
        else:
            # If cycle column is not found, assume the last 90 days is the current cycle
            lookback_days = 90
            current_date = cycles_data.index[-1]
            start_date = current_date - pd.Timedelta(days=lookback_days)
            current_cycle_data = cycles_data[cycles_data.index >= start_date]

        # Extract historical cycles
        historical_cycles = {}

        if cycle_column in cycles_data.columns:
            for cycle_id in cycles_data[cycle_column].unique():
                if cycle_id != current_cycle_id:  # Skip the current cycle
                    cycle_data = cycles_data[cycles_data[cycle_column] == cycle_id]
                    if len(cycle_data) > 0:
                        historical_cycles[str(cycle_id)] = cycle_data

        # If no historical cycles found or cycle column not present, create them by year
        if not historical_cycles:
            # Group by year as a fallback
            for year in set(cycles_data.index.year):
                if year != cycles_data.index[-1].year:  # Skip current year
                    year_data = cycles_data[cycles_data.index.year == year]
                    if len(year_data) > 0:
                        historical_cycles[str(year)] = year_data

        # Prepare data for comparison
        comparison_results = {}

        # Check if we have enough data for comparison
        if len(current_cycle_data) < 10 or not historical_cycles:
            return {"error": "Insufficient data for comparison", "similarity_scores": {}}

        # Extract price data for comparison
        current_prices = current_cycle_data['close'].values

        # If normalize is True, normalize the current prices
        if normalize:
            current_prices = current_prices / current_prices[0]

        # Compare with each historical cycle
        similarity_scores = {}

        for cycle_id, cycle_data in historical_cycles.items():
            historical_prices = cycle_data['close'].values

            # Skip if not enough data points
            if len(historical_prices) < len(current_prices):
                continue

            # If normalize is True, normalize the historical prices
            if normalize:
                historical_prices = historical_prices / historical_prices[0]

            # Calculate similarity using Dynamic Time Warping (DTW)
            # We'll use Euclidean distance as a simple measure here
            # DTW would be more accurate but requires additional libraries

            # For each possible starting point in the historical data
            min_distance = float('inf')
            best_start_idx = 0

            for start_idx in range(len(historical_prices) - len(current_prices) + 1):
                segment = historical_prices[start_idx:start_idx + len(current_prices)]

                # Calculate Euclidean distance
                distance = np.sqrt(np.sum((segment - current_prices) ** 2))

                if distance < min_distance:
                    min_distance = distance
                    best_start_idx = start_idx

            # Convert to similarity score (higher is better)
            similarity = 1 / (1 + min_distance)

            # Store the results
            similarity_scores[cycle_id] = {
                "similarity": similarity,
                "start_idx": best_start_idx,
                "matched_length": len(current_prices)
            }

        # Sort cycles by similarity
        sorted_scores = {k: v for k, v in sorted(
            similarity_scores.items(),
            key=lambda item: item[1]["similarity"],
            reverse=True
        )}

        # Get the most similar cycle
        most_similar_cycle_id = next(iter(sorted_scores), None)
        most_similar_cycle_data = None

        if most_similar_cycle_id:
            most_similar_cycle_data = historical_cycles[most_similar_cycle_id]
            best_start_idx = sorted_scores[most_similar_cycle_id]["start_idx"]
            matched_length = sorted_scores[most_similar_cycle_id]["matched_length"]

            # Calculate potential continuation based on the most similar historical cycle
            current_length = len(current_cycle_data)
            if len(most_similar_cycle_data) > best_start_idx + matched_length:
                # Extract the continuation segment
                continuation_segment = most_similar_cycle_data.iloc[best_start_idx + matched_length:]

                # If normalized, adjust the continuation to match the current price level
                if normalize:
                    adjustment_factor = current_prices[-1] / (
                                most_similar_cycle_data['close'].iloc[best_start_idx + matched_length - 1] /
                                most_similar_cycle_data['close'].iloc[best_start_idx])
                    predicted_continuation = continuation_segment['close'].values * adjustment_factor
                else:
                    adjustment_factor = current_prices[-1] / most_similar_cycle_data['close'].iloc[
                        best_start_idx + matched_length - 1]
                    predicted_continuation = continuation_segment['close'].values * adjustment_factor

        # Prepare the return dictionary
        comparison_results = {
            "similarity_scores": sorted_scores,
            "current_cycle_length": len(current_cycle_data),
            "current_cycle_start": current_cycle_data.index[0].strftime('%Y-%m-%d'),
            "current_cycle_end": current_cycle_data.index[-1].strftime('%Y-%m-%d'),
            "most_similar_cycle": most_similar_cycle_id if most_similar_cycle_id else None,
        }

        # Add potential continuation data if available
        if most_similar_cycle_id and 'predicted_continuation' in locals():
            comparison_results["potential_continuation"] = {
                "prices": predicted_continuation.tolist(),
                "duration_days": len(continuation_segment),
                "based_on_cycle": most_similar_cycle_id,
                "confidence_score": sorted_scores[most_similar_cycle_id]["similarity"]
            }

        return comparison_results

    def predict_cycle_turning_points(self, processed_data: pd.DataFrame,
                                     symbol: str,
                                     cycle_type: str = 'auto',
                                     confidence_interval: float = 0.9) -> pd.DataFrame:

        symbol = symbol.upper().replace('USDT', '').replace('USD', '')

        # Determine the appropriate cycle type if 'auto' is specified
        if cycle_type == 'auto':
            if symbol == 'BTC':
                cycle_type = 'halving'
            elif symbol == 'ETH':
                cycle_type = 'network_upgrade'
            elif symbol == 'SOL':
                cycle_type = 'ecosystem_event'
            else:
                cycle_type = 'bull_bear'

        # Ensure the DataFrame has a datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Create a copy of the processed data
        df = processed_data.copy()

        # Get historical cycles comparison for context
        cycle_comparison = self.compare_current_to_historical_cycles(
            processed_data=df,
            symbol=symbol,
            cycle_type=cycle_type,
            normalize=True
        )

        # Initialize technical indicators for turning point detection
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Calculate volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()

        # Initialize result dataframe for turning points
        turning_points = pd.DataFrame(columns=[
            'date', 'price', 'direction', 'strength', 'confidence', 'indicators',
            'days_since_last_tp', 'cycle_phase'
        ])

        # Define functions to detect turning point signals
        def detect_potential_bottom(row, prev_rows):
            signals = []
            strength = 0

            # RSI oversold condition
            if row['rsi'] < 30:
                signals.append('RSI_oversold')
                strength += 1

                # RSI bullish divergence (price making lower low but RSI making higher low)
                if (len(prev_rows) >= 10 and
                        row['close'] < prev_rows['close'].min() and
                        row['rsi'] > prev_rows['rsi'].min()):
                    signals.append('RSI_bullish_divergence')
                    strength += 2

            # MACD bullish crossover
            if row['macd'] > row['macd_signal'] and prev_rows['macd'].iloc[-2] <= prev_rows['macd_signal'].iloc[-2]:
                signals.append('MACD_bullish_cross')
                strength += 1

            # MACD bullish divergence
            if (len(prev_rows) >= 10 and
                    row['close'] < prev_rows['close'].min() and
                    row['macd'] > prev_rows['macd'].min()):
                signals.append('MACD_bullish_divergence')
                strength += 2

            # Moving average support
            if (abs(row['close'] - row['sma_200']) / row['sma_200'] < 0.03 and
                    row['close'] < row['sma_200']):
                signals.append('MA200_support')
                strength += 1

            # Volume spike
            if 'volume' in row and row['volume'] > prev_rows['volume'].mean() * 2:
                signals.append('volume_spike')
                strength += 1

            return signals, strength

        def detect_potential_top(row, prev_rows):
            signals = []
            strength = 0

            # RSI overbought condition
            if row['rsi'] > 70:
                signals.append('RSI_overbought')
                strength += 1

                # RSI bearish divergence (price making higher high but RSI making lower high)
                if (len(prev_rows) >= 10 and
                        row['close'] > prev_rows['close'].max() and
                        row['rsi'] < prev_rows['rsi'].max()):
                    signals.append('RSI_bearish_divergence')
                    strength += 2

            # MACD bearish crossover
            if row['macd'] < row['macd_signal'] and prev_rows['macd'].iloc[-2] >= prev_rows['macd_signal'].iloc[-2]:
                signals.append('MACD_bearish_cross')
                strength += 1

            # MACD bearish divergence
            if (len(prev_rows) >= 10 and
                    row['close'] > prev_rows['close'].max() and
                    row['macd'] < prev_rows['macd'].max()):
                signals.append('MACD_bearish_divergence')
                strength += 2

            # Moving average resistance
            if (abs(row['close'] - row['sma_200']) / row['sma_200'] < 0.03 and
                    row['close'] > row['sma_200']):
                signals.append('MA200_resistance')
                strength += 1

            # Volume spike with price rejection
            if ('volume' in row and
                    'high' in row and 'low' in row and 'open' in row and 'close' in row and
                    row['volume'] > prev_rows['volume'].mean() * 2 and
                    row['close'] < row['open'] and
                    (row['high'] - max(row['open'], row['close'])) > (min(row['open'], row['close']) - row['low'])):
                signals.append('volume_spike_rejection')
                strength += 1

            return signals, strength

        # Get cycle-specific information based on cycle_type
        if cycle_type == 'halving' and symbol == 'BTC':
            # For BTC halving cycles, add halving-specific indicators
            if 'halving_cycle_phase' in df.columns:
                # Historical analysis shows tops tend to occur 300-500 days after halving
                df['days_since_last_halving'] = df['days_since_last_halving'].fillna(0)
                df['halving_top_probability'] = (
                    np.exp(-0.5 * ((df['days_since_last_halving'] - 400) / 100) ** 2)
                )

                # Historical analysis shows bottoms tend to occur 100-200 days before halving
                df['days_to_next_halving'] = df['days_to_next_halving'].fillna(2000)
                df['halving_bottom_probability'] = (
                    np.exp(-0.5 * ((df['days_to_next_halving'] - 150) / 50) ** 2)
                )

        # Define lookback window for turning point detection
        lookback_window = 30

        # Process each data point, skipping the first lookback_window points
        for i in range(lookback_window, len(df)):
            row = df.iloc[i]
            prev_rows = df.iloc[i - lookback_window:i]

            # Detect potential bottom
            bottom_signals, bottom_strength = detect_potential_bottom(row, prev_rows)

            # Detect potential top
            top_signals, top_strength = detect_potential_top(row, prev_rows)

            # Adjust strength based on cycle-specific factors
            if cycle_type == 'halving' and symbol == 'BTC' and 'halving_top_probability' in df.columns:
                top_strength *= (1 + row['halving_top_probability'])
                bottom_strength *= (1 + row.get('halving_bottom_probability', 0))

            # Determine if this is a significant turning point
            if bottom_signals and bottom_strength >= 3:
                # Calculate days since last turning point
                if len(turning_points) > 0:
                    days_since_last_tp = (row.name - turning_points['date'].iloc[-1]).days
                else:
                    days_since_last_tp = None

                # Calculate confidence based on strength and other factors
                confidence = min(0.95, bottom_strength / 6)

                # Determine cycle phase
                if 'market_phase' in row:
                    cycle_phase = row['market_phase']
                elif 'halving_cycle_phase' in row:
                    cycle_phase = f"Halving cycle: {row['halving_cycle_phase']:.2f}"
                else:
                    cycle_phase = "Unknown"

                # Add to turning points dataframe
                turning_points = pd.concat([turning_points, pd.DataFrame([{
                    'date': row.name,
                    'price': row['close'],
                    'direction': 'bottom',
                    'strength': bottom_strength,
                    'confidence': confidence,
                    'indicators': ', '.join(bottom_signals),
                    'days_since_last_tp': days_since_last_tp,
                    'cycle_phase': cycle_phase
                }])], ignore_index=True)

            elif top_signals and top_strength >= 3:
                # Calculate days since last turning point
                if len(turning_points) > 0:
                    days_since_last_tp = (row.name - turning_points['date'].iloc[-1]).days
                else:
                    days_since_last_tp = None

                # Calculate confidence based on strength and other factors
                confidence = min(0.95, top_strength / 6)

                # Determine cycle phase
                if 'market_phase' in row:
                    cycle_phase = row['market_phase']
                elif 'halving_cycle_phase' in row:
                    cycle_phase = f"Halving cycle: {row['halving_cycle_phase']:.2f}"
                else:
                    cycle_phase = "Unknown"

                # Add to turning points dataframe
                turning_points = pd.concat([turning_points, pd.DataFrame([{
                    'date': row.name,
                    'price': row['close'],
                    'direction': 'top',
                    'strength': top_strength,
                    'confidence': confidence,
                    'indicators': ', '.join(top_signals),
                    'days_since_last_tp': days_since_last_tp,
                    'cycle_phase': cycle_phase
                }])], ignore_index=True)

        # Filter turning points by the confidence interval
        turning_points = turning_points[turning_points['confidence'] >= confidence_interval]

        # Use historical cycle comparison to predict future turning points
        if 'potential_continuation' in cycle_comparison:
            potential_prices = cycle_comparison['potential_continuation']['prices']

            if len(potential_prices) > 10:
                # Calculate the last known date and project future dates
                last_date = df.index[-1]
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, len(potential_prices) + 1)]

                # Create a DataFrame with projected prices
                projected_df = pd.DataFrame({
                    'close': potential_prices,
                    'date': future_dates
                })
                projected_df.set_index('date', inplace=True)

                # Calculate basic metrics for the projected data
                projected_df['price_change'] = projected_df['close'].pct_change()

                # Detect potential future turning points
                lookback = min(10, len(projected_df))
                for i in range(lookback, len(projected_df)):
                    curr_price = projected_df['close'].iloc[i]
                    prev_prices = projected_df['close'].iloc[i - lookback:i]

                    # Simple peak detection
                    if all(curr_price > prev_prices):
                        # Possible future top
                        confidence = min(0.8, cycle_comparison['potential_continuation']['confidence_score'])

                        turning_points = pd.concat([turning_points, pd.DataFrame([{
                            'date': projected_df.index[i],
                            'price': curr_price,
                            'direction': 'projected_top',
                            'strength': 3,  # Default value
                            'confidence': confidence,
                            'indicators': 'Historical pattern projection',
                            'days_since_last_tp': (projected_df.index[i] - turning_points['date'].iloc[-1]).days if len(
                                turning_points) > 0 else None,
                            'cycle_phase': f"Projected ({cycle_comparison['most_similar_cycle']})"
                        }])], ignore_index=True)

                    # Simple valley detection
                    elif all(curr_price < prev_prices):
                        # Possible future bottom
                        confidence = min(0.8, cycle_comparison['potential_continuation']['confidence_score'])

                        turning_points = pd.concat([turning_points, pd.DataFrame([{
                            'date': projected_df.index[i],
                            'price': curr_price,
                            'direction': 'projected_bottom',
                            'strength': 3,  # Default value
                            'confidence': confidence,
                            'indicators': 'Historical pattern projection',
                            'days_since_last_tp': (projected_df.index[i] - turning_points['date'].iloc[-1]).days if len(
                                turning_points) > 0 else None,
                            'cycle_phase': f"Projected ({cycle_comparison['most_similar_cycle']})"
                        }])], ignore_index=True)

        # Sort turning points by date
        turning_points = turning_points.sort_values('date')

        # For BTC halving cycles, add known future halving dates as significant turning points
        if symbol == 'BTC' and cycle_type == 'halving':
            current_date = df.index[-1]

            for halving_date_str in self.btc_halving_dates:
                halving_date = pd.Timestamp(halving_date_str)

                # Only add future halving dates
                if halving_date > current_date:
                    turning_points = pd.concat([turning_points, pd.DataFrame([{
                        'date': halving_date,
                        'price': None,  # Unknown future price
                        'direction': 'halving_event',
                        'strength': 5,  # High significance
                        'confidence': 0.99,  # Very high confidence for the event (though not for price)
                        'indicators': 'Bitcoin halving event',
                        'days_since_last_tp': None,
                        'cycle_phase': 'Halving event'
                    }])], ignore_index=True)

        # For ETH and SOL, add upcoming network events if available
        elif symbol == 'ETH' and cycle_type == 'network_upgrade':
            current_date = df.index[-1]

            for event in self.eth_significant_events:
                event_date = pd.Timestamp(event['date'])

                # Only add future events
                if event_date > current_date:
                    turning_points = pd.concat([turning_points, pd.DataFrame([{
                        'date': event_date,
                        'price': None,  # Unknown future price
                        'direction': 'network_event',
                        'strength': 4,  # High significance
                        'confidence': 0.9,  # High confidence for the event
                        'indicators': f"Ethereum {event['name']}: {event['description']}",
                        'days_since_last_tp': None,
                        'cycle_phase': 'Network upgrade'
                    }])], ignore_index=True)

        elif symbol == 'SOL' and cycle_type == 'ecosystem_event':
            current_date = df.index[-1]

            for event in self.sol_significant_events:
                event_date = pd.Timestamp(event['date'])

                # Only add future events
                if event_date > current_date:
                    turning_points = pd.concat([turning_points, pd.DataFrame([{
                        'date': event_date,
                        'price': None,  # Unknown future price
                        'direction': 'ecosystem_event',
                        'strength': 4,  # High significance
                        'confidence': 0.9,  # High confidence for the event
                        'indicators': f"Solana {event['name']}: {event['description']}",
                        'days_since_last_tp': None,
                        'cycle_phase': 'Ecosystem event'
                    }])], ignore_index=True)

        # Sort by date again after adding events
        turning_points = turning_points.sort_values('date')

        return turning_points



    def analyze_token_correlations(self, symbols: List[str],
                                   timeframe: str = '1d',
                                   lookback_period: str = '1 year') -> pd.DataFrame:

        # Calculate end date (current date) and start date based on lookback period
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Parse the lookback period
        value, unit = lookback_period.split()
        value = int(value)

        if 'year' in unit:
            start_date = (datetime.now() - timedelta(days=365 * value)).strftime('%Y-%m-%d')
        elif 'month' in unit:
            start_date = (datetime.now() - timedelta(days=30 * value)).strftime('%Y-%m-%d')
        elif 'week' in unit:
            start_date = (datetime.now() - timedelta(weeks=value)).strftime('%Y-%m-%d')
        elif 'day' in unit:
            start_date = (datetime.now() - timedelta(days=value)).strftime('%Y-%m-%d')
        else:
            raise ValueError(f"Invalid lookback period format: {lookback_period}")

        # Dictionary to store price data for each symbol
        price_data = {}
        returns_data = {}

        # Load data for each symbol
        for symbol in symbols:
            # Load processed data
            data = self.load_processed_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if len(data) == 0:
                print(f"No data available for {symbol}. Skipping.")
                continue

            # Store close prices
            price_data[symbol] = data['close']

            # Calculate daily returns
            returns_data[symbol] = data['close'].pct_change().fillna(0)

        # Create DataFrames
        prices_df = pd.DataFrame(price_data)
        returns_df = pd.DataFrame(returns_data)

        # Calculate correlation matrices
        price_correlation = prices_df.corr()
        returns_correlation = returns_df.corr()

        # Create a more detailed correlation analysis
        result = {
            'price_correlation': price_correlation,
            'returns_correlation': returns_correlation
        }

        # Analyze correlations during different market phases
        # We'll use BTC as a reference for market phases if available
        if 'BTC' in symbols or 'BTCUSDT' in symbols:
            btc_symbol = 'BTC' if 'BTC' in symbols else 'BTCUSDT'
            btc_data = self.load_processed_data(
                symbol=btc_symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # Detect market phases
            btc_phases = self.detect_market_phase(btc_data)

            # Create phase-specific correlation matrices
            for phase in ['accumulation', 'uptrend', 'distribution', 'downtrend']:
                phase_dates = btc_phases[btc_phases['market_phase'] == phase].index

                if len(phase_dates) > 0:
                    # Filter returns for this phase
                    phase_returns = returns_df.loc[phase_dates].dropna(how='all')

                    if len(phase_returns) > 1:  # Need at least 2 data points for correlation
                        phase_corr = phase_returns.corr()
                        result[f'{phase}_correlation'] = phase_corr

        # Convert to DataFrame with multi-level columns for better output
        correlation_types = list(result.keys())
        symbols_list = list(price_data.keys())

        # Create empty DataFrame with multi-level columns
        multi_idx = pd.MultiIndex.from_product([correlation_types, symbols_list],
                                               names=['correlation_type', 'symbol'])
        final_df = pd.DataFrame(index=symbols_list, columns=multi_idx)

        # Fill the DataFrame
        for corr_type in correlation_types:
            if corr_type in result:
                for sym1 in symbols_list:
                    for sym2 in symbols_list:
                        if sym1 in result[corr_type].index and sym2 in result[corr_type].columns:
                            final_df.loc[sym1, (corr_type, sym2)] = result[corr_type].loc[sym1, sym2]

        # Save to database
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        correlation_data = {
            'timestamp': timestamp,
            'lookback_period': lookback_period,
            'timeframe': timeframe,
            'symbols': symbols,
            'correlation_matrices': result
        }

        # Save correlation data to the database
        # This is just a placeholder - actual implementation would depend on your DB schema
        # self.db_connection.save_correlation_analysis(correlation_data)

        return final_df

    def update_features_with_new_data(self, processed_data: pd.DataFrame,
                                      symbol: str) -> pd.DataFrame:

        if len(processed_data) == 0:
            raise ValueError("No data provided for update.")

        # Ensure we have a datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Clean symbol
        symbol_clean = symbol.upper().replace('USDT', '').replace('USD', '')

        # Get the last date in the processed data
        last_date = processed_data.index[-1]

        # Retrieve the existing cycle features from the database
        # This is a placeholder - actual implementation would depend on your DB schema
        existing_features = self.db_connection.get_latest_cycle_features(symbol=symbol)

        # If no existing features, process all data
        if existing_features is None or len(existing_features) == 0:
            return self.create_cyclical_features(processed_data, symbol)

        # Determine which specific updates are needed based on the symbol
        updated_features = processed_data.copy()

        # Update general market phase detection
        updated_features = self.detect_market_phase(updated_features)

        # Update bull/bear cycle identification
        updated_features = self.identify_bull_bear_cycles(updated_features)

        # Update token-specific cycle features
        if symbol_clean == 'BTC':
            updated_features = self.calculate_btc_halving_cycle_features(updated_features)
        elif symbol_clean == 'ETH':
            updated_features = self.calculate_eth_event_cycle_features(updated_features)
        elif symbol_clean == 'SOL':
            updated_features = self.calculate_sol_event_cycle_features(updated_features)

        # Check for new cycle turning points
        turning_points = self.predict_cycle_turning_points(updated_features, symbol)

        # Save any new turning points to the database
        if turning_points is not None and len(turning_points) > 0:
            # This is a placeholder - actual implementation depends on your DB schema
            for _, point in turning_points.iterrows():
                self.db_connection.save_predicted_turning_point(
                    symbol=symbol,
                    date=point['date'],
                    point_type=point['type'],
                    confidence=point['confidence'],
                    description=point['description']
                )

        # Check for cycle anomalies
        anomalies = self.detect_cycle_anomalies(updated_features, symbol)

        # Add anomaly information to features
        if anomalies is not None and len(anomalies) > 0:
            for col in anomalies.columns:
                if col not in updated_features.columns:
                    updated_features[col] = anomalies[col]

        # Compare to historical cycles
        historical_comparison = self.compare_current_to_historical_cycles(updated_features, symbol)

        # Save the comparison results
        if historical_comparison:
            # This is a placeholder - actual implementation depends on your DB schema
            for ref_cycle, similarity in historical_comparison.get('similarities', {}).items():
                self.db_connection.save_cycle_similarity(
                    symbol=symbol,
                    reference_cycle=ref_cycle,
                    current_cycle=historical_comparison.get('current_cycle'),
                    similarity_score=similarity,
                    date=last_date
                )

        # Save the updated features to the database
        # This is a placeholder - actual implementation depends on your DB schema
        self.db_connection.save_cycle_feature(
            symbol=symbol,
            timeframe=processed_data.attrs.get('timeframe', '1d'),
            features=updated_features,
            date=last_date
        )

        return updated_features