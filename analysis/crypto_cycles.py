"""
Enhanced Crypto Cycles Analysis Module

This module provides functionality to analyze cryptocurrency market cycles,
including Bitcoin halving cycles, Ethereum network upgrades, Solana ecosystem events,
bull/bear market detection, and other cyclical patterns that can be used
as features for deep learning models.

This class is designed to work with pre-processed data rather than raw
candle data to improve efficiency and performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

# Import from project modules
from models.time_series import detect_seasonality
from data.db import DatabaseManager
from utils.crypto_helpers import calculate_drawdown, calculate_roi
from analysis.trend_detection import detect_trend_change_points

""""
функції для роботи з базою даних
get_klines_processed
save_market_cycle
update_market_cycle
get_market_cycle_by_id
get_market_cycles_by_symbol
get_active_market_cycles
delete_market_cycle
save_cycle_feature
get_cycle_features
get_latest_cycle_features
delete_cycle_feature
save_cycle_similarity
get_cycle_similarities_by_reference
get_most_similar_cycles
save_predicted_turning_point
update_turning_point_outcome
get_pending_turning_points
get_turning_points_by_date_range
insert_cycle_feature_performance
get_cycle_feature_performance
"""
class CryptoCycles:
    """
    A class for analyzing cryptocurrency market cycles to enhance prediction models.
    Serves as a helper class for deep_learning.py by providing additional cyclical
    features that can improve deep learning model performance.

    Enhanced with support for ETH and SOL specific cycles and events.
    """

    def __init__(self):
        """
        Initialize the CryptoCycles analyzer.

        Parameters:
        -----------
        db_connection : connection object, optional
            Database connection for retrieving historical data.
            If None, a new connection will be created when needed.
        """
        self.db_connection = DatabaseManager()

        # Bitcoin halving dates
        self.btc_halving_dates = [
            "2012-11-28",  # First halving
            "2016-07-09",  # Second halving
            "2020-05-11",  # Third halving
            "2024-04-20",  # Estimated fourth halving
        ]

        # Ethereum significant network upgrades/events
        self.eth_significant_events = [
            {"date": "2015-07-30", "name": "Frontier", "description": "Initial launch"},
            {"date": "2016-03-14", "name": "Homestead", "description": "First planned upgrade"},
            {"date": "2016-07-20", "name": "DAO Fork", "description": "Hard fork after DAO hack"},
            {"date": "2017-10-16", "name": "Byzantium", "description": "First part of Metropolis upgrade"},
            {"date": "2019-02-28", "name": "Constantinople", "description": "Second part of Metropolis upgrade"},
            {"date": "2019-12-08", "name": "Istanbul", "description": "Final hard fork before ETH 2.0"},
            {"date": "2020-12-01", "name": "Beacon Chain", "description": "Launch of ETH 2.0 Phase 0"},
            {"date": "2021-08-05", "name": "London", "description": "EIP-1559 implementation"},
            {"date": "2022-09-15", "name": "The Merge", "description": "Transition to Proof of Stake"},
            {"date": "2023-04-12", "name": "Shanghai", "description": "Enabled staking withdrawals"},
            {"date": "2023-03-16", "name": "Capella", "description": "Enabled validator withdrawals"},
            {"date": "2024-03-13", "name": "Dencun", "description": "Proto-danksharding with blobs"}
        ]

        # Solana significant events
        self.sol_significant_events = [
            {"date": "2020-03-16", "name": "Mainnet Beta", "description": "Initial launch of Solana mainnet beta"},
            {"date": "2021-05-26", "name": "Wormhole", "description": "Cross-chain bridge to Ethereum launched"},
            {"date": "2021-09-14", "name": "Network Outage", "description": "Major network outage lasting 17 hours"},
            {"date": "2022-02-02", "name": "Wormhole Hack", "description": "$320M Wormhole hack"},
            {"date": "2022-06-01", "name": "Network Outage", "description": "4-hour network outage"},
            {"date": "2023-02-25", "name": "Network Outage", "description": "20-hour outage due to validator issues"},
            {"date": "2023-06-23", "name": "Network Update", "description": "QUIC implementation to prevent spam"},
            {"date": "2024-02-06", "name": "Firedancer", "description": "Introduction of alternate client"}
        ]

        # Dictionary to map symbol to its significant events
        self.symbol_events_map = {
            "BTC": self.btc_halving_dates,
            "ETH": self.eth_significant_events,
            "SOL": self.sol_significant_events
        }

        self.cached_processed_data = {}

    def load_processed_data(self, symbol: str, timeframe: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load pre-processed price data from the database or cache.
        """
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"

        if cache_key in self.cached_processed_data:
            return self.cached_processed_data[cache_key]

        # Load pre-processed data from storage manager
        processed_data = self.db_connection.get_klines_processed(
            symbol=symbol,
            interval=timeframe,
            start_date=start_date,
            end_date=end_date,
            connection=self.db_connection
        )

        # Cache for future use
        self.cached_processed_data[cache_key] = processed_data

        return processed_data

    def detect_market_phase(self, processed_data: pd.DataFrame, window: int = 30) -> pd.DataFrame:

        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = processed_data.copy()

        # Ensure we have the required columns
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in result_df.columns:
                raise ValueError(f"Required column '{col}' not found in processed_data")

        # Calculate additional technical indicators
        # 1. SMA for trend direction
        result_df['sma_short'] = result_df['close'].rolling(window=window // 3).mean()
        result_df['sma_long'] = result_df['close'].rolling(window=window).mean()

        # 2. Volume trend
        result_df['volume_sma'] = result_df['volume'].rolling(window=window // 2).mean()

        # 3. Price volatility
        result_df['price_std'] = result_df['close'].rolling(window=window // 2).std()
        result_df['volatility'] = result_df['price_std'] / result_df['close']

        # 4. Price momentum
        result_df['momentum'] = result_df['close'].pct_change(periods=window // 4)

        # 5. RSI for overbought/oversold conditions
        delta = result_df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
        result_df['rsi'] = 100 - (100 / (1 + rs))

        # Initialize market phase column
        result_df['market_phase'] = 'undefined'

        # Define market phases based on indicators
        # Phase 1: Accumulation
        accum_mask = (
                (result_df['sma_short'] < result_df['sma_long']) &
                (result_df['momentum'] > -0.05) &
                (result_df['momentum'] < 0.05) &
                (result_df['volatility'] < result_df['volatility'].rolling(window=window * 2).mean()) &
                (result_df['rsi'] < 50)
        )
        result_df.loc[accum_mask, 'market_phase'] = 'accumulation'

        # Phase 2: Uptrend
        uptrend_mask = (
                (result_df['sma_short'] > result_df['sma_long']) &
                (result_df['momentum'] > 0) &
                (result_df['volume'] > result_df['volume_sma'])
        )
        result_df.loc[uptrend_mask, 'market_phase'] = 'uptrend'

        # Phase 3: Distribution
        distrib_mask = (
                (result_df['sma_short'] > result_df['sma_long']) &
                (result_df['momentum'] < 0.05) &
                (result_df['momentum'] > -0.05) &
                (result_df['volatility'] < result_df['volatility'].rolling(window=window * 2).mean()) &
                (result_df['rsi'] > 50)
        )
        result_df.loc[distrib_mask, 'market_phase'] = 'distribution'

        # Phase 4: Downtrend
        downtrend_mask = (
                (result_df['sma_short'] < result_df['sma_long']) &
                (result_df['momentum'] < 0)
        )
        result_df.loc[downtrend_mask, 'market_phase'] = 'downtrend'

        # Remove intermediate calculation columns to keep the DataFrame clean
        # Comment this out if you want to keep these columns for debugging
        columns_to_drop = ['sma_short', 'sma_long', 'volume_sma', 'price_std',
                           'volatility', 'momentum', 'rsi']
        result_df = result_df.drop(columns=columns_to_drop, errors='ignore')

        # Forward fill any NaN values in the market_phase column
        result_df['market_phase'] = result_df['market_phase'].ffill()

        return result_df

    def identify_bull_bear_cycles(self, processed_data: pd.DataFrame,
                                  threshold_bull: float = 0.2,
                                  threshold_bear: float = -0.2,
                                  min_duration_days: int = 20) -> pd.DataFrame:

        # Create a copy of the input DataFrame
        df = processed_data.copy()

        # Ensure we have the required columns
        if 'close' not in df.columns:
            raise ValueError("Required column 'close' not found in processed_data")

        # Ensure the DataFrame has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Calculate rolling maximum and minimum prices
        df['rolling_max'] = df['close'].cummax()
        df['drawdown'] = (df['close'] / df['rolling_max']) - 1

        df['rolling_min'] = df['close'].cummin()
        df['recovery'] = (df['close'] / df['rolling_min']) - 1

        # Initialize cycle state variables
        cycle_states = pd.DataFrame(index=df.index)
        cycle_states['cycle_state'] = 'undefined'
        cycle_states['cycle_id'] = 0
        cycle_states['days_in_cycle'] = 0
        cycle_states['cycle_start_price'] = np.nan
        cycle_states['cycle_max_price'] = np.nan
        cycle_states['cycle_min_price'] = np.nan
        cycle_states['cycle_current_roi'] = 0.0
        cycle_states['cycle_max_roi'] = 0.0
        cycle_states['cycle_max_drawdown'] = 0.0

        # Identify cycle states
        current_state = 'undefined'
        current_cycle_id = 0
        cycle_start_idx = 0
        cycle_start_price = df['close'].iloc[0]
        max_price_in_current_cycle = df['close'].iloc[0]
        min_price_in_current_cycle = df['close'].iloc[0]

        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            current_date = df.index[i]

            # Check if we need to change state
            if current_state == 'undefined' or current_state == 'bear':
                # Check for bull market signal
                if df['recovery'].iloc[i] >= threshold_bull:
                    if current_state != 'bull':
                        # Start a new bull cycle
                        current_state = 'bull'
                        current_cycle_id += 1
                        cycle_start_idx = i
                        cycle_start_price = current_price
                        max_price_in_current_cycle = current_price
                        min_price_in_current_cycle = current_price

            elif current_state == 'bull':
                # Check for bear market signal
                if df['drawdown'].iloc[i] <= threshold_bear:
                    # Switch to bear cycle
                    current_state = 'bear'
                    current_cycle_id += 1
                    cycle_start_idx = i
                    cycle_start_price = current_price
                    max_price_in_current_cycle = current_price
                    min_price_in_current_cycle = current_price

            # Update cycle statistics
            days_in_cycle = (df.index[i] - df.index[cycle_start_idx]).days

            # Update max and min prices in the current cycle
            max_price_in_current_cycle = max(max_price_in_current_cycle, current_price)
            min_price_in_current_cycle = min(min_price_in_current_cycle, current_price)

            # Calculate ROI and drawdown for the current cycle
            cycle_current_roi = (current_price / cycle_start_price) - 1
            cycle_max_roi = (max_price_in_current_cycle / cycle_start_price) - 1
            cycle_max_drawdown = (min_price_in_current_cycle / max_price_in_current_cycle) - 1

            # Update the cycle states DataFrame
            cycle_states.loc[current_date, 'cycle_state'] = current_state
            cycle_states.loc[current_date, 'cycle_id'] = current_cycle_id
            cycle_states.loc[current_date, 'days_in_cycle'] = days_in_cycle
            cycle_states.loc[current_date, 'cycle_start_price'] = cycle_start_price
            cycle_states.loc[current_date, 'cycle_max_price'] = max_price_in_current_cycle
            cycle_states.loc[current_date, 'cycle_min_price'] = min_price_in_current_cycle
            cycle_states.loc[current_date, 'cycle_current_roi'] = cycle_current_roi
            cycle_states.loc[current_date, 'cycle_max_roi'] = cycle_max_roi
            cycle_states.loc[current_date, 'cycle_max_drawdown'] = cycle_max_drawdown

        # Filter out cycles that don't meet the minimum duration
        valid_cycles = []
        for cycle_id in cycle_states['cycle_id'].unique():
            cycle_data = cycle_states[cycle_states['cycle_id'] == cycle_id]
            if cycle_data['days_in_cycle'].max() >= min_duration_days:
                valid_cycles.append(cycle_id)

        # Mark invalid cycles as undefined
        cycle_states.loc[~cycle_states['cycle_id'].isin(valid_cycles), 'cycle_state'] = 'undefined'

        # Create a summary DataFrame with cycle statistics
        cycles_summary = []

        for cycle_id in valid_cycles:
            cycle_data = cycle_states[cycle_states['cycle_id'] == cycle_id]
            if len(cycle_data) > 0:
                cycle_type = cycle_data['cycle_state'].iloc[0]
                start_date = cycle_data.index[0]
                end_date = cycle_data.index[-1]
                duration_days = (end_date - start_date).days
                start_price = cycle_data['cycle_start_price'].iloc[0]
                end_price = df.loc[end_date, 'close']
                max_price = cycle_data['cycle_max_price'].max()
                min_price = cycle_data['cycle_min_price'].min()
                total_roi = (end_price / start_price) - 1
                max_roi = cycle_data['cycle_max_roi'].max()
                max_drawdown = cycle_data['cycle_max_drawdown'].min()

                cycle_summary = {
                    'cycle_id': cycle_id,
                    'cycle_type': cycle_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': duration_days,
                    'start_price': start_price,
                    'end_price': end_price,
                    'max_price': max_price,
                    'min_price': min_price,
                    'total_roi': total_roi,
                    'max_roi': max_roi,
                    'max_drawdown': max_drawdown
                }
                cycles_summary.append(cycle_summary)

        # Create the cycles summary DataFrame
        cycles_summary_df = pd.DataFrame(cycles_summary)

        # Merge the original data with cycle information
        result = pd.concat([df, cycle_states], axis=1)

        # Remove intermediate calculation columns
        result = result.drop(['rolling_max', 'drawdown', 'rolling_min', 'recovery'], axis=1, errors='ignore')

        # Add cycles_summary as an attribute
        result.cycles_summary = cycles_summary_df if len(cycles_summary) > 0 else pd.DataFrame()

        return result

    def calculate_btc_halving_cycle_features(self, processed_data: pd.DataFrame) -> pd.DataFrame:

        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = processed_data.copy()

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Convert halving dates to datetime objects
        halving_dates = [pd.Timestamp(date) for date in self.btc_halving_dates]

        # Add next estimated halving date if it's not already included
        # Bitcoin halvings occur approximately every 210,000 blocks (~ 4 years)
        if len(halving_dates) > 0:
            last_halving = halving_dates[-1]
            next_halving = last_halving + pd.DateOffset(days=1461)  # ~4 years = 1461 days

            # Only add the next estimated halving if it's not already included
            if next_halving > halving_dates[-1]:
                halving_dates.append(next_halving)

        # Initialize cycle features
        result_df['days_since_last_halving'] = None
        result_df['days_to_next_halving'] = None
        result_df['halving_cycle_phase'] = None
        result_df['cycle_number'] = None

        # Calculate cycle features for each date in the DataFrame
        for idx, date in enumerate(result_df.index):
            # Find the previous and next halving dates
            previous_halving = None
            next_halving = None
            cycle_number = 0

            for i, halving_date in enumerate(halving_dates):
                if date >= halving_date:
                    previous_halving = halving_date
                    cycle_number = i + 1  # Cycle number starts from 1
                else:
                    next_halving = halving_date
                    break

            # Calculate days since last halving
            if previous_halving is not None:
                days_since_last_halving = (date - previous_halving).days
                result_df.at[date, 'days_since_last_halving'] = days_since_last_halving
            else:
                # If no previous halving, set to NaN
                result_df.at[date, 'days_since_last_halving'] = np.nan

            # Calculate days to next halving
            if next_halving is not None:
                days_to_next_halving = (next_halving - date).days
                result_df.at[date, 'days_to_next_halving'] = days_to_next_halving
            else:
                # If no next halving in our list, estimate based on 4-year cycle
                if previous_halving is not None:
                    estimated_next_halving = previous_halving + pd.DateOffset(days=1461)
                    days_to_next_halving = (estimated_next_halving - date).days
                    result_df.at[date, 'days_to_next_halving'] = days_to_next_halving
                else:
                    result_df.at[date, 'days_to_next_halving'] = np.nan

            # Calculate halving cycle phase (0-1 value representing position in cycle)
            if previous_halving is not None and next_halving is not None:
                cycle_length = (next_halving - previous_halving).days
                days_into_cycle = (date - previous_halving).days
                cycle_phase = days_into_cycle / cycle_length
                result_df.at[date, 'halving_cycle_phase'] = cycle_phase
            elif previous_halving is not None:
                # If we're in the last known cycle, estimate based on 4-year cycle
                estimated_next_halving = previous_halving + pd.DateOffset(days=1461)
                cycle_length = 1461  # ~4 years in days
                days_into_cycle = (date - previous_halving).days
                cycle_phase = days_into_cycle / cycle_length
                result_df.at[date, 'halving_cycle_phase'] = cycle_phase
            else:
                result_df.at[date, 'halving_cycle_phase'] = np.nan

            # Set cycle number
            result_df.at[date, 'cycle_number'] = cycle_number

        # Convert features to appropriate data types
        result_df['days_since_last_halving'] = result_df['days_since_last_halving'].astype('float64')
        result_df['days_to_next_halving'] = result_df['days_to_next_halving'].astype('float64')
        result_df['halving_cycle_phase'] = result_df['halving_cycle_phase'].astype('float64')
        result_df['cycle_number'] = result_df['cycle_number'].astype('int64')

        # Add additional derived features

        # Log-transformed days since last halving (useful for machine learning)
        result_df['log_days_since_halving'] = np.log1p(result_df['days_since_last_halving'])

        # Sine and cosine transformation for cyclical features (better for neural networks)
        cycle_phase = result_df['halving_cycle_phase'] * 2 * np.pi
        result_df['halving_cycle_sin'] = np.sin(cycle_phase)
        result_df['halving_cycle_cos'] = np.cos(cycle_phase)

        # Relative price change since halving
        # This requires grouping by cycle and calculating percent change from cycle start
        for cycle in result_df['cycle_number'].unique():
            cycle_mask = result_df['cycle_number'] == cycle
            if cycle_mask.any():
                cycle_start_price = result_df.loc[cycle_mask, 'close'].iloc[0]
                result_df.loc[cycle_mask, 'price_change_since_halving'] = (
                        result_df.loc[cycle_mask, 'close'] / cycle_start_price - 1
                )

        return result_df

    def calculate_eth_event_cycle_features(self, processed_data: pd.DataFrame) -> pd.DataFrame:

        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = processed_data.copy()

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Convert Ethereum events to datetime objects and sort them
        eth_events = sorted(self.eth_significant_events, key=lambda x: pd.Timestamp(x["date"]))
        eth_event_dates = [pd.Timestamp(event["date"]) for event in eth_events]

        # Add known future Ethereum upgrades if they exist
        # For example, add placeholder for next upcoming upgrade if officially announced
        # This would normally come from a database or external source
        next_known_upgrade = None
        # Check if there's an officially announced next upgrade that's not in our list
        # This logic would be replaced with actual lookup logic in a real implementation

        # Initialize features
        result_df['days_since_last_upgrade'] = None
        result_df['days_to_next_known_upgrade'] = None
        result_df['upgrade_cycle_phase'] = None
        result_df['eth2_phase_indicator'] = 0  # ETH 2.0 phase indicator (0-4)
        result_df['pos_transition_indicator'] = 0  # Proof of Stake transition progress (0-1)

        # The Merge date (transition to PoS)
        merge_date = pd.Timestamp("2022-09-15")

        # Beacon Chain Launch date (start of ETH 2.0 phase 0)
        beacon_chain_date = pd.Timestamp("2020-12-01")

        # ETH 2.0 phases with approximate dates
        eth2_phases = [
            {"phase": 0, "date": pd.Timestamp("2020-12-01"), "name": "Beacon Chain"},  # Phase 0
            {"phase": 1, "date": pd.Timestamp("2022-09-15"), "name": "The Merge"},  # Phase 1
            {"phase": 2, "date": pd.Timestamp("2023-04-12"), "name": "Shanghai/Capella"},  # Phase 2
            {"phase": 3, "date": pd.Timestamp("2024-03-13"), "name": "Dencun"}  # Phase 3
            # Phase 4 would be future sharding upgrades or other major changes
        ]

        # Calculate features for each date in the DataFrame
        for idx, date in enumerate(result_df.index):
            # Find the previous and next upgrade dates
            previous_upgrade = None
            previous_upgrade_name = None
            next_upgrade = None
            next_upgrade_name = None

            for i, event in enumerate(eth_events):
                event_date = pd.Timestamp(event["date"])
                if date >= event_date:
                    previous_upgrade = event_date
                    previous_upgrade_name = event["name"]
                else:
                    next_upgrade = event_date
                    next_upgrade_name = event["name"]
                    break

            # Calculate days since last upgrade
            if previous_upgrade is not None:
                days_since_last_upgrade = (date - previous_upgrade).days
                result_df.at[date, 'days_since_last_upgrade'] = days_since_last_upgrade
            else:
                result_df.at[date, 'days_since_last_upgrade'] = np.nan

            # Calculate days to next known upgrade
            if next_upgrade is not None:
                days_to_next_upgrade = (next_upgrade - date).days
                result_df.at[date, 'days_to_next_known_upgrade'] = days_to_next_upgrade
            elif next_known_upgrade is not None:
                # Use announced future upgrade if available
                days_to_next_upgrade = (next_known_upgrade - date).days
                result_df.at[date, 'days_to_next_known_upgrade'] = days_to_next_upgrade
            else:
                result_df.at[date, 'days_to_next_known_upgrade'] = np.nan

            # Calculate upgrade cycle phase (0-1 value representing position in cycle)
            if previous_upgrade is not None and next_upgrade is not None:
                cycle_length = (next_upgrade - previous_upgrade).days
                days_into_cycle = (date - previous_upgrade).days
                cycle_phase = days_into_cycle / cycle_length if cycle_length > 0 else 0
                result_df.at[date, 'upgrade_cycle_phase'] = cycle_phase
            else:
                # If we can't determine cycle phase, set to NaN
                result_df.at[date, 'upgrade_cycle_phase'] = np.nan

            # Calculate ETH 2.0 phase indicator
            # This is a number (0-4) representing which ETH 2.0 phase we're in
            eth2_phase = 0
            for phase_info in eth2_phases:
                if date >= phase_info["date"]:
                    eth2_phase = phase_info["phase"] + 1  # +1 because phases are 0-indexed
            result_df.at[date, 'eth2_phase_indicator'] = eth2_phase

            # Calculate PoS transition indicator
            # Before Beacon Chain: 0
            # Between Beacon Chain and Merge: value between 0-1 based on progress
            # After Merge: 1
            if date < beacon_chain_date:
                result_df.at[date, 'pos_transition_indicator'] = 0
            elif date >= merge_date:
                result_df.at[date, 'pos_transition_indicator'] = 1
            else:
                # Linear progression between Beacon Chain and Merge
                total_transition_days = (merge_date - beacon_chain_date).days
                days_since_beacon = (date - beacon_chain_date).days
                transition_progress = days_since_beacon / total_transition_days if total_transition_days > 0 else 0
                result_df.at[date, 'pos_transition_indicator'] = transition_progress

        # Convert features to appropriate data types
        result_df['days_since_last_upgrade'] = result_df['days_since_last_upgrade'].astype('float64')
        result_df['days_to_next_known_upgrade'] = result_df['days_to_next_known_upgrade'].astype('float64')
        result_df['upgrade_cycle_phase'] = result_df['upgrade_cycle_phase'].astype('float64')
        result_df['eth2_phase_indicator'] = result_df['eth2_phase_indicator'].astype('int64')
        result_df['pos_transition_indicator'] = result_df['pos_transition_indicator'].astype('float64')

        # Add additional derived features

        # Log-transformed days since last upgrade (useful for machine learning)
        result_df['log_days_since_upgrade'] = np.log1p(result_df['days_since_last_upgrade'])

        # Sine and cosine transformation for cyclical features (better for neural networks)
        cycle_phase = result_df['upgrade_cycle_phase'] * 2 * np.pi
        result_df['upgrade_cycle_sin'] = np.sin(cycle_phase)
        result_df['upgrade_cycle_cos'] = np.cos(cycle_phase)

        # Add upgrade importance weight based on historical price impact
        # This is a simplified implementation - in reality you'd want to analyze
        # historical price movements around each upgrade
        result_df['upgrade_importance'] = 0.0

        upgrade_importance = {
            "The Merge": 1.0,  # Most significant
            "London": 0.8,  # EIP-1559 was very significant
            "Beacon Chain": 0.7,  # Initial ETH 2.0 launch
            "Shanghai": 0.6,  # Enabled withdrawals
            "Dencun": 0.6,  # Proto-danksharding
            "Constantinople": 0.4,  # Medium impact
            "Byzantium": 0.4,
            "Istanbul": 0.3,
            "Homestead": 0.3,
            "Frontier": 0.2,
            "DAO Fork": 0.9,  # Very significant but for negative reasons
            "Capella": 0.5  # Significant for stakers
        }

        # Apply importance values based on the most recent upgrade
        for idx, date in enumerate(result_df.index):
            for event in eth_events:
                event_date = pd.Timestamp(event["date"])
                if date >= event_date:
                    importance = upgrade_importance.get(event["name"], 0.3)  # Default to 0.3
                    result_df.at[date, 'upgrade_importance'] = importance

        return result_df

    def calculate_sol_event_cycle_features(self, processed_data: pd.DataFrame) -> pd.DataFrame:

        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = processed_data.copy()

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Convert Solana events to datetime objects and sort them
        sol_events = sorted(self.sol_significant_events, key=lambda x: pd.Timestamp(x["date"]))
        sol_event_dates = [pd.Timestamp(event["date"]) for event in sol_events]

        # Classify events by type
        outage_events = [event for event in sol_events if "Outage" in event["name"]]
        upgrade_events = [event for event in sol_events if "Update" in event["name"] or "Firedancer" in event["name"]]
        ecosystem_events = [event for event in sol_events if "Mainnet" in event["name"] or "Wormhole" in event["name"]]

        # Create a weighted dictionary for outage impact (higher values mean more severe outages)
        outage_impact = {
            pd.Timestamp("2021-09-14"): 0.9,  # 17-hour outage was very severe
            pd.Timestamp("2022-06-01"): 0.5,  # 4-hour outage
            pd.Timestamp("2023-02-25"): 0.8,  # 20-hour outage
        }

        # Initialize features
        result_df['days_since_last_significant_event'] = None
        result_df['days_since_last_outage'] = None
        result_df['network_stability_score'] = None
        result_df['ecosystem_growth_phase'] = None

        # Define ecosystem growth phases
        # Phase 0: Pre-launch
        # Phase 1: Initial launch (2020-03-16 to 2021-05-26)
        # Phase 2: Early growth (2021-05-26 to 2022-02-02)
        # Phase 3: Stabilization period (2022-02-02 to 2023-06-23)
        # Phase 4: Maturity phase (2023-06-23 onwards)
        growth_phase_dates = [
            pd.Timestamp("2020-03-16"),  # Phase 1 start
            pd.Timestamp("2021-05-26"),  # Phase 2 start
            pd.Timestamp("2022-02-02"),  # Phase 3 start
            pd.Timestamp("2023-06-23"),  # Phase 4 start
        ]

        # Calculate features for each date in the DataFrame
        for idx, date in enumerate(result_df.index):
            # Find the previous significant event
            previous_event = None
            previous_event_name = None

            for i, event in enumerate(sol_events):
                event_date = pd.Timestamp(event["date"])
                if date >= event_date:
                    previous_event = event_date
                    previous_event_name = event["name"]
                else:
                    break

            # Calculate days since last significant event
            if previous_event is not None:
                days_since_event = (date - previous_event).days
                result_df.at[date, 'days_since_last_significant_event'] = days_since_event
            else:
                result_df.at[date, 'days_since_last_significant_event'] = np.nan

            # Calculate days since last outage
            last_outage_date = None
            for event in outage_events:
                event_date = pd.Timestamp(event["date"])
                if date >= event_date:
                    last_outage_date = event_date
                else:
                    break

            if last_outage_date is not None:
                days_since_outage = (date - last_outage_date).days
                result_df.at[date, 'days_since_last_outage'] = days_since_outage

                # Decay function: impact decreases over time
                # We use the exponential decay formula: impact * exp(-lambda * t)
                # where t is time in days and lambda determines decay speed
                decay_factor = 0.01  # Smaller value means slower decay
                outage_date_impact = outage_impact.get(last_outage_date, 0.5)  # Default impact
                stability_score = 1 - (outage_date_impact * np.exp(-decay_factor * days_since_outage))

                # The score approaches 1.0 over time after an outage
                result_df.at[date, 'network_stability_score'] = stability_score
            else:
                result_df.at[date, 'days_since_last_outage'] = np.nan
                result_df.at[date, 'network_stability_score'] = 1.0  # Perfect score if no outages

            # Calculate ecosystem growth phase (integer 0-4)
            growth_phase = 0
            for i, phase_date in enumerate(growth_phase_dates):
                if date >= phase_date:
                    growth_phase = i + 1
            result_df.at[date, 'ecosystem_growth_phase'] = growth_phase

        # Convert features to appropriate data types
        result_df['days_since_last_significant_event'] = result_df['days_since_last_significant_event'].astype(
            'float64')
        result_df['days_since_last_outage'] = result_df['days_since_last_outage'].astype('float64')
        result_df['network_stability_score'] = result_df['network_stability_score'].astype('float64')
        result_df['ecosystem_growth_phase'] = result_df['ecosystem_growth_phase'].astype('int64')

        # Add additional derived features

        # Log-transformed days since last event (useful for machine learning)
        result_df['log_days_since_event'] = np.log1p(result_df['days_since_last_significant_event'])

        # Calculate time-weighted ecosystem maturity score (0-1)
        # This is a continuous measure of ecosystem maturity based on time since launch
        sol_launch_date = pd.Timestamp("2020-03-16")
        max_maturity_days = 1095  # ~3 years to reach "maturity"

        for idx, date in enumerate(result_df.index):
            if date >= sol_launch_date:
                days_since_launch = (date - sol_launch_date).days
                maturity_score = min(1.0, days_since_launch / max_maturity_days)
                result_df.at[date, 'ecosystem_maturity_score'] = maturity_score
            else:
                result_df.at[date, 'ecosystem_maturity_score'] = 0.0

        # Add network growth indicators
        # This would ideally be based on actual metrics like daily active addresses,
        # transactions per day, etc., but we'll use a simulated value based on phases
        for idx, date in enumerate(result_df.index):
            growth_phase = result_df.at[date, 'ecosystem_growth_phase']

            # Base growth multiplier based on phase
            if growth_phase == 0:
                growth_mult = 0
            elif growth_phase == 1:
                growth_mult = 0.2
            elif growth_phase == 2:
                growth_mult = 0.5
            elif growth_phase == 3:
                growth_mult = 0.8
            else:  # Phase 4
                growth_mult = 1.0

            # Adjust for known network issues
            if result_df.at[date, 'network_stability_score'] < 0.7:
                growth_mult *= 0.7  # Reduce growth during periods of instability

            result_df.at[date, 'network_growth_indicator'] = growth_mult

        return result_df

    def calculate_token_specific_cycle_features(self,
                                                processed_data: pd.DataFrame,
                                                symbol: str) -> pd.DataFrame:
        """
        Calculate token-specific cycle features based on the cryptocurrency symbol.
        This serves as a router to the appropriate cycle calculation method.

        Parameters:
        -----------
        processed_data : pd.DataFrame
            Pre-processed DataFrame containing price data with datetime index.
        symbol : str
            Cryptocurrency symbol ('BTC', 'ETH', 'SOL', etc.)

        Returns:
        --------
        pd.DataFrame
            Original DataFrame with additional token-specific cycle features.
        """
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

    def analyze_weekly_cycle(self, processed_data: pd.DataFrame) -> Dict:

        # Ensure we have the required data
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        if 'close' not in processed_data.columns:
            raise ValueError("DataFrame must have a 'close' column")

        # Create a copy to avoid modifying the original data
        df = processed_data.copy()

        # Extract day of week
        df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

        # Analyze returns by day of week
        daily_returns = df['close'].pct_change().fillna(0)
        df['daily_return'] = daily_returns

        # Calculate statistics by day of week
        day_stats = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for day_num, day_name in enumerate(day_names):
            day_returns = df[df['day_of_week'] == day_num]['daily_return']

            if len(day_returns) > 0:
                day_stats[day_name] = {
                    'mean_return': day_returns.mean(),
                    'median_return': day_returns.median(),
                    'positive_days': (day_returns > 0).sum() / len(day_returns),
                    'negative_days': (day_returns < 0).sum() / len(day_returns),
                    'volatility': day_returns.std(),
                    'max_gain': day_returns.max(),
                    'max_loss': day_returns.min(),
                    'sample_size': len(day_returns)
                }

        # Calculate weekly momentum patterns
        # Identify if there are patterns like "Monday dip" or "Weekend effect"
        week_momentum = {}

        # Check if specific days tend to reverse the previous day's trend
        for i in range(1, 7):  # Tuesday through Sunday
            prev_day = i - 1
            current_day = i

            prev_day_data = df[df['day_of_week'] == prev_day]['daily_return']
            current_day_data = df[df['day_of_week'] == current_day]['daily_return']

            # Create pairs of previous day and current day returns
            # Need to align dates properly
            prev_dates = df[df['day_of_week'] == prev_day].index
            next_day_dates = [date + pd.Timedelta(days=1) for date in prev_dates]

            pairs = []
            for date in prev_dates:
                next_date = date + pd.Timedelta(days=1)
                if next_date in df.index:
                    prev_return = df.loc[date, 'daily_return']
                    next_return = df.loc[next_date, 'daily_return']
                    pairs.append((prev_return, next_return))

            if pairs:
                prev_returns = [p[0] for p in pairs]
                next_returns = [p[1] for p in pairs]

                # Calculate correlation between previous and current day returns
                correlation = np.corrcoef(prev_returns, next_returns)[0, 1] if len(pairs) > 1 else 0

                # Calculate how often the current day continues or reverses previous day trend
                continuation = sum(1 for p in pairs if (p[0] > 0 and p[1] > 0) or (p[0] < 0 and p[1] < 0))
                reversal = sum(1 for p in pairs if (p[0] > 0 and p[1] < 0) or (p[0] < 0 and p[1] > 0))

                total_pairs = len(pairs)

                week_momentum[f"{day_names[prev_day]} to {day_names[current_day]}"] = {
                    'correlation': correlation,
                    'continuation_rate': continuation / total_pairs if total_pairs > 0 else 0,
                    'reversal_rate': reversal / total_pairs if total_pairs > 0 else 0,
                    'sample_size': total_pairs
                }

        # Calculate average week pattern (useful for visualization)
        avg_week_pattern = {}
        for day_num, day_name in enumerate(day_names):
            avg_week_pattern[day_name] = df[df['day_of_week'] == day_num]['close'].mean()

        # Return the compiled statistics
        return {
            'day_of_week_stats': day_stats,
            'weekly_momentum_patterns': week_momentum,
            'average_week_pattern': avg_week_pattern,
            'best_day': max(day_stats.items(), key=lambda x: x[1]['mean_return'])[0],
            'worst_day': min(day_stats.items(), key=lambda x: x[1]['mean_return'])[0],
            'most_volatile_day': max(day_stats.items(), key=lambda x: x[1]['volatility'])[0],
            'weekend_effect': {
                'fri_to_mon_correlation': week_momentum.get('Friday to Monday', {}).get('correlation', 0),
                'weekend_to_weekday_return_ratio':
                    (day_stats.get('Saturday', {}).get('mean_return', 0) +
                     day_stats.get('Sunday', {}).get('mean_return', 0)) /
                    sum(day_stats.get(d, {}).get('mean_return', 0) for d in
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
                    if sum(day_stats.get(d, {}).get('mean_return', 0) for d in
                           ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) != 0 else 0
            }
        }

    def analyze_monthly_seasonality(self, processed_data: pd.DataFrame, years_back: int = 3) -> Dict:

        # Ensure we have the required data
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        if 'close' not in processed_data.columns:
            raise ValueError("DataFrame must have a 'close' column")

        # Create a copy to avoid modifying the original data
        df = processed_data.copy()

        # Filter data based on years_back parameter
        cutoff_date = df.index.max() - pd.DateOffset(years=years_back)
        df = df[df.index >= cutoff_date]

        # Extract month and year
        df['month'] = df.index.month
        df['year'] = df.index.year

        # Calculate monthly returns
        df['monthly_return'] = df.groupby(['year', 'month'])['close'].transform(
            lambda x: (x.iloc[-1] / x.iloc[0]) - 1 if len(x) > 0 else np.nan
        )

        # Calculate statistics by month
        month_stats = {}
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        for month_num, month_name in enumerate(month_names, 1):
            month_data = df[df['month'] == month_num]
            month_returns = []

            # Calculate return for each month instance
            for year in month_data['year'].unique():
                year_month_data = month_data[month_data['year'] == year]
                if len(year_month_data) > 0:
                    first_price = year_month_data['close'].iloc[0]
                    last_price = year_month_data['close'].iloc[-1]
                    month_return = (last_price / first_price) - 1
                    month_returns.append(month_return)

            if month_returns:
                month_stats[month_name] = {
                    'mean_return': np.mean(month_returns),
                    'median_return': np.median(month_returns),
                    'positive_months': sum(1 for r in month_returns if r > 0) / len(month_returns),
                    'negative_months': sum(1 for r in month_returns if r < 0) / len(month_returns),
                    'volatility': np.std(month_returns),
                    'max_gain': max(month_returns),
                    'max_loss': min(month_returns),
                    'sample_size': len(month_returns),
                    'returns_by_year': {year: return_value for year, return_value in
                                        zip(df[df['month'] == month_num]['year'].unique(), month_returns)}
                }

        # Calculate quarterly statistics
        quarters = {
            'Q1': [1, 2, 3],  # Jan-Mar
            'Q2': [4, 5, 6],  # Apr-Jun
            'Q3': [7, 8, 9],  # Jul-Sep
            'Q4': [10, 11, 12]  # Oct-Dec
        }

        quarter_stats = {}

        for quarter_name, months in quarters.items():
            quarter_data = df[df['month'].isin(months)]
            quarter_returns = []

            # Calculate return for each quarter instance
            for year in quarter_data['year'].unique():
                year_quarter_data = quarter_data[quarter_data['year'] == year]
                if len(year_quarter_data) > 0:
                    # Find first and last dates in this quarter for this year
                    first_date = year_quarter_data.index.min()
                    last_date = year_quarter_data.index.max()

                    first_price = df.loc[first_date, 'close']
                    last_price = df.loc[last_date, 'close']

                    quarter_return = (last_price / first_price) - 1
                    quarter_returns.append(quarter_return)

            if quarter_returns:
                quarter_stats[quarter_name] = {
                    'mean_return': np.mean(quarter_returns),
                    'median_return': np.median(quarter_returns),
                    'positive_quarters': sum(1 for r in quarter_returns if r > 0) / len(quarter_returns),
                    'negative_quarters': sum(1 for r in quarter_returns if r < 0) / len(quarter_returns),
                    'volatility': np.std(quarter_returns),
                    'max_gain': max(quarter_returns),
                    'max_loss': min(quarter_returns),
                    'sample_size': len(quarter_returns)
                }

        # Check for January effect (if January is positive, full year tends to be positive)
        january_effect = {}
        years_with_january = df[df['month'] == 1]['year'].unique()

        for year in years_with_january:
            jan_data = df[(df['year'] == year) & (df['month'] == 1)]
            full_year_data = df[df['year'] == year]

            if len(jan_data) > 0 and len(full_year_data) > 0:
                jan_return = (jan_data['close'].iloc[-1] / jan_data['close'].iloc[0]) - 1
                year_return = (full_year_data['close'].iloc[-1] / full_year_data['close'].iloc[0]) - 1

                january_effect[year] = {
                    'january_return': jan_return,
                    'full_year_return': year_return,
                    'january_predicted_year': (jan_return > 0 and year_return > 0) or (
                                jan_return < 0 and year_return < 0)
                }

        # Calculate January effect accuracy
        if january_effect:
            jan_effect_accuracy = sum(1 for v in january_effect.values() if v['january_predicted_year']) / len(
                january_effect)
        else:
            jan_effect_accuracy = 0

        # Check for seasonality significance using bootstrapping or other statistical methods
        # This is a simple implementation - in practice, you might want to use more sophisticated methods

        # Calculate mean returns for each month across years
        mean_returns_by_month = {month: stats['mean_return'] for month, stats in month_stats.items()}

        # Calculate standard deviation of the monthly mean returns
        if mean_returns_by_month:
            monthly_std = np.std(list(mean_returns_by_month.values()))
            # Z-scores to identify significantly different months (abs(z) > 1.96 for p < 0.05)
            mean_of_means = np.mean(list(mean_returns_by_month.values()))
            z_scores = {month: (mean - mean_of_means) / monthly_std if monthly_std > 0 else 0
                        for month, mean in mean_returns_by_month.items()}

            significant_months = {month: score for month, score in z_scores.items() if abs(score) > 1.96}
        else:
            monthly_std = 0
            z_scores = {}
            significant_months = {}

        # Return the compiled statistics
        return {
            'monthly_stats': month_stats,
            'quarterly_stats': quarter_stats,
            'best_month': max(month_stats.items(), key=lambda x: x[1]['mean_return'])[0] if month_stats else None,
            'worst_month': min(month_stats.items(), key=lambda x: x[1]['mean_return'])[0] if month_stats else None,
            'most_volatile_month': max(month_stats.items(), key=lambda x: x[1]['volatility'])[
                0] if month_stats else None,
            'january_effect': {
                'accuracy': jan_effect_accuracy,
                'yearly_data': january_effect
            },
            'seasonality_significance': {
                'monthly_z_scores': z_scores,
                'significant_months': significant_months,
                'seasonality_strength': len(significant_months) / len(month_stats) if month_stats else 0
            },
            'years_analyzed': years_back,
            'data_timespan': {
                'start_date': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
                'end_date': df.index.max().strftime('%Y-%m-%d') if not df.empty else None
            }
        }

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

    def calculate_volatility_by_cycle_phase(self, processed_data: pd.DataFrame,
                                          symbol: str,
                                          cycle_type: str = 'auto',
                                          window: int = 14) -> pd.DataFrame:
        """
        Calculate volatility metrics for different cycle phases.

        Parameters:
        -----------
        processed_data : pd.DataFrame
            Pre-processed DataFrame containing price data with datetime index.
        symbol : str
            Cryptocurrency symbol to determine which cycle definitions to use.
        cycle_type : str, default='auto'
            Type of cycle to analyze (automatic selection based on symbol).
        window : int, default=14
            Window size for volatility calculation.

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility metrics for different cycle phases.
        """
        # Implementation with symbol-specific logic would go here
        pass

    def detect_cycle_anomalies(self, processed_data: pd.DataFrame,
                              symbol: str,
                              cycle_type: str = 'auto') -> pd.DataFrame:
        """
        Detect anomalies in current cycle compared to historical cycles.

        Parameters:
        -----------
        processed_data : pd.DataFrame
            Pre-processed DataFrame containing price data with datetime index.
        symbol : str
            Cryptocurrency symbol to determine which cycle definitions to use.
        cycle_type : str, default='auto'
            Type of cycle to analyze (automatic selection based on symbol).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing detected anomalies with their significance scores.
        """
        # Implementation with symbol-specific logic would go here
        pass

    def get_significant_events_for_symbol(self, symbol: str) -> List:
        """
        Get the list of significant events for a specific cryptocurrency.

        Parameters:
        -----------
        symbol : str
            Cryptocurrency symbol ('BTC', 'ETH', 'SOL', etc.)

        Returns:
        --------
        List
            List of significant events for the specified symbol.
        """
        symbol = symbol.upper().replace('USDT', '').replace('USD', '')
        return self.symbol_events_map.get(symbol, [])

    def compare_current_to_historical_cycles(self, processed_data: pd.DataFrame,
                                           symbol: str,
                                           cycle_type: str = 'auto',
                                           normalize: bool = True) -> Dict:
        """
        Compare the current market cycle to historical cycles using pattern matching.

        Parameters:
        -----------
        processed_data : pd.DataFrame
            Pre-processed DataFrame containing price data with datetime index.
        symbol : str
            Cryptocurrency symbol to analyze.
        cycle_type : str, default='auto'
            Type of cycle to compare (automatic selection based on symbol).
        normalize : bool, default=True
            Whether to normalize prices for comparison.

        Returns:
        --------
        Dict
            Dictionary containing similarity scores and closest historical cycles.
        """
        # Implementation with symbol-specific logic would go here
        pass

    def predict_cycle_turning_points(self, processed_data: pd.DataFrame,
                                    symbol: str,
                                    cycle_type: str = 'auto',
                                    confidence_interval: float = 0.9) -> pd.DataFrame:
        """
        Predict potential turning points in the current market cycle.

        Parameters:
        -----------
        processed_data : pd.DataFrame
            Pre-processed DataFrame containing price data with datetime index.
        symbol : str
            Cryptocurrency symbol to analyze.
        cycle_type : str, default='auto'
            Type of cycle to analyze for turning points (auto-selected based on symbol).
        confidence_interval : float, default=0.9
            Confidence interval for predictions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing predicted turning points with probabilities.
        """
        # Implementation with symbol-specific logic would go here
        pass

    def plot_cycle_comparison(self, processed_data: pd.DataFrame,
                             symbol: str,
                             cycle_type: str = 'auto',
                             normalize: bool = True,
                             save_path: Optional[str] = None) -> None:
        """
        Plot comparison of current cycle against historical cycles.

        Parameters:
        -----------
        processed_data : pd.DataFrame
            Pre-processed DataFrame containing price data with datetime index.
        symbol : str
            Cryptocurrency symbol to analyze.
        cycle_type : str, default='auto'
            Type of cycle to compare (auto-selected based on symbol).
        normalize : bool, default=True
            Whether to normalize prices for comparison.
        save_path : str, optional
            If provided, save the plot to this path.
        """
        # Implementation with symbol-specific logic would go here
        pass

    def analyze_token_correlations(self, symbols: List[str],
                                  timeframe: str = '1d',
                                  lookback_period: str = '1 year') -> pd.DataFrame:
        """
        Analyze correlations between different tokens across their respective cycles.

        Parameters:
        -----------
        symbols : List[str]
            List of cryptocurrency symbols to analyze (e.g., ['BTC', 'ETH', 'SOL']).
        timeframe : str, default='1d'
            Timeframe for the analysis.
        lookback_period : str, default='1 year'
            Period to look back for correlation analysis.

        Returns:
        --------
        pd.DataFrame
            Correlation matrix between different tokens' price movements.
        """
        # Implementation would go here
        pass

    def extract_cycle_features_for_deep_learning(self, symbol: str,
                                              timeframe: str = '1d',
                                              lookback_period: Optional[str] = '5 years',
                                              include_all_symbols: bool = False) -> pd.DataFrame:
        """
        Extract comprehensive cycle-related features for deep learning models.

        This is the main interface method for integration with deep_learning.py.

        Parameters:
        -----------
        symbol : str
            Primary cryptocurrency symbol to analyze.
        timeframe : str, default='1d'
            Timeframe for the analysis ('1d', '4h', etc.).
        lookback_period : str, optional
            Period to look back for data, e.g., '5 years', '2 years'.
            If None, will use all available data.
        include_all_symbols : bool, default=False
            Whether to include cycle features for all available symbols.

        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted cycle features ready for deep learning models.
        """
        # Calculate end date (current date) and start date based on lookback period
        end_date = datetime.now().strftime('%Y-%m-%d')

        if lookback_period:
            # Parse the lookback period
            value, unit = lookback_period.split()
            value = int(value)

            if 'year' in unit:
                start_date = (datetime.now() - timedelta(days=365*value)).strftime('%Y-%m-%d')
            elif 'month' in unit:
                start_date = (datetime.now() - timedelta(days=30*value)).strftime('%Y-%m-%d')
            elif 'week' in unit:
                start_date = (datetime.now() - timedelta(weeks=value)).strftime('%Y-%m-%d')
            elif 'day' in unit:
                start_date = (datetime.now() - timedelta(days=value)).strftime('%Y-%m-%d')
            else:
                start_date = None
        else:
            start_date = None

        # Load processed data
        processed_data = self.load_processed_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Create general and symbol-specific cyclical features
        result_df = self.create_cyclical_features(processed_data, symbol)

        # If requested, include features from other symbols
        if include_all_symbols:
            # Get correlation features with other major cryptocurrencies
            for additional_symbol in ['BTC', 'ETH', 'SOL']:
                if additional_symbol != symbol:
                    # Load data for additional symbol
                    additional_data = self.load_processed_data(
                        symbol=additional_symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )

                    # Calculate correlation features and join to main dataframe
                    # Implementation would go here
                    pass

        return result_df

    def update_features_with_new_data(self, processed_data: pd.DataFrame,
                                    symbol: str) -> pd.DataFrame:
        """
        Update cycle features with newly processed data.

        This method is designed to be called when new processed data becomes available,
        to efficiently update the cycle features without reprocessing all historical data.

        Parameters:
        -----------
        processed_data : pd.DataFrame
            New pre-processed DataFrame containing price data with datetime index.
        symbol : str
            Cryptocurrency symbol to analyze.

        Returns:
        --------
        pd.DataFrame
            DataFrame with updated cycle features.
        """
        # Implementation with symbol-specific logic would go here
        pass