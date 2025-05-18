import numpy as np
import pandas as pd
from utils.config import *
from utils.logger import CryptoLogger
from decimal import Decimal


class BitcoinCycleFeatureExtractor:
    def __init__(self):
        self.btc_halving_dates = btc_halving_dates
        self.logger = CryptoLogger('BitcoinCycleFeatureExtractor')

    def calculate_btc_halving_cycle_features(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting calculation of Bitcoin halving cycle features")

        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = processed_data.copy()
        self.logger.debug(f"Input dataframe shape: {result_df.shape}")

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.error("DataFrame index is not a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Convert halving dates to datetime objects
        halving_dates = [pd.Timestamp(date) for date in self.btc_halving_dates]
        self.logger.info(f"Using halving dates: {halving_dates}")

        # Add next estimated halving date if it's not already included
        # Bitcoin halvings occur approximately every 210,000 blocks (~ 4 years)
        if len(halving_dates) > 0:
            last_halving = halving_dates[-1]
            next_halving = last_halving + pd.DateOffset(days=1461)  # ~4 years = 1461 days
            self.logger.info(f"Last known halving: {last_halving}, estimated next halving: {next_halving}")

            # Only add the next estimated halving if it's not already included
            if next_halving > halving_dates[-1]:
                halving_dates.append(next_halving)
                self.logger.debug(f"Added next estimated halving date: {next_halving}")

        # Initialize cycle features
        result_df['days_since_last_halving'] = None
        result_df['days_to_next_halving'] = None
        result_df['halving_cycle_phase'] = None
        result_df['cycle_number'] = None

        self.logger.info(f"Processing {len(result_df)} data points for halving cycle features")

        # Calculate cycle features for each date in the DataFrame
        for idx, date in enumerate(result_df.index):
            if idx % 1000 == 0:
                self.logger.debug(f"Processing date {idx}/{len(result_df)}: {date}")

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
        self.logger.debug("Converting feature data types")
        result_df['days_since_last_halving'] = result_df['days_since_last_halving'].astype('float64')
        result_df['days_to_next_halving'] = result_df['days_to_next_halving'].astype('float64')
        result_df['halving_cycle_phase'] = result_df['halving_cycle_phase'].astype('float64')
        result_df['cycle_number'] = result_df['cycle_number'].astype('int64')

        # Add additional derived features
        self.logger.info("Calculating additional derived features")

        # Log-transformed days since last halving (useful for machine learning)
        result_df['log_days_since_halving'] = np.log1p(result_df['days_since_last_halving'])

        # Sine and cosine transformation for cyclical features (better for neural networks)
        cycle_phase = result_df['halving_cycle_phase'] * 2 * np.pi
        result_df['halving_cycle_sin'] = np.sin(cycle_phase)
        result_df['halving_cycle_cos'] = np.cos(cycle_phase)

        # Relative price change since halving
        # This requires grouping by cycle and calculating percent change from cycle start
        self.logger.info("Calculating price change since halving for each cycle")
        for cycle in result_df['cycle_number'].unique():
            cycle_mask = result_df['cycle_number'] == cycle
            if cycle_mask.any():
                try:
                    # Convert to the same data type (float) before division to avoid Decimal/float incompatibility
                    cycle_start_price = float(result_df.loc[cycle_mask, 'close'].iloc[0])

                    # Convert all 'close' values to float within this cycle
                    close_values = result_df.loc[cycle_mask, 'close'].astype(float)

                    # Calculate price change
                    price_change = (close_values / cycle_start_price) - 1

                    result_df.loc[cycle_mask, 'price_change_since_halving'] = price_change

                    self.logger.debug(f"Processed cycle {cycle} with starting price {cycle_start_price}")

                except Exception as e:
                    self.logger.error(f"Error calculating price change for cycle {cycle}: {e}")
                    self.logger.error(f"Cycle start price type: {type(result_df.loc[cycle_mask, 'close'].iloc[0])}")
                    # Set to NaN for this cycle
                    result_df.loc[cycle_mask, 'price_change_since_halving'] = np.nan

        self.logger.info("Bitcoin halving cycle features calculation completed")
        return result_df