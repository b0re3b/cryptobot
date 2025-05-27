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
        """
            Обчислює додаткові ознаки, пов'язані з циклами халвінгу Bitcoin, на основі вхідних часових рядів.

            Цей метод додає до DataFrame інформацію про:
            - скільки днів минуло з останнього халвінгу,
            - скільки днів залишилось до наступного халвінгу,
            - фазу халвінгового циклу (нормалізовану від 0 до 1),
            - номер циклу,
            - логарифм часу з останнього халвінгу,
            - синус і косинус фази циклу (для кращої роботи моделей машинного навчання),
            - відносну зміну ціни від початку поточного халвінгу.

            Args:
                processed_data (pd.DataFrame): Вхідний DataFrame з часовими рядами цін Bitcoin.
                    Повинен містити колонку з датою в індексі (або у стовпці 'date' чи 'timestamp'),
                    а також колонку 'close' з цінами закриття.

            Returns:
                pd.DataFrame: Копія вхідного DataFrame з доданими ознаками халвінгових циклів:
                    - 'days_since_last_halving' (float): Кількість днів від останнього халвінгу.
                    - 'days_to_next_halving' (float): Кількість днів до наступного халвінгу.
                    - 'halving_cycle_phase' (float): Нормалізована фаза халвінгового циклу (0–1).
                    - 'cycle_number' (int): Номер халвінгового циклу (починаючи з 1).
                    - 'log_days_since_halving' (float): Логарифм (1 + days_since_last_halving).
                    - 'halving_cycle_sin' (float): Синус фази циклу (для моделювання циклічності).
                    - 'halving_cycle_cos' (float): Косинус фази циклу.
                    - 'price_change_since_halving' (float): Відносна зміна ціни від початку циклу.


            """
        self.logger.info("Starting calculation of Bitcoin halving cycle features")

        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = processed_data.copy()
        self.logger.debug(f"Input dataframe shape: {result_df.shape}")
        self.logger.debug(f"Current index type: {type(result_df.index)}")

        # Handle different index types and convert to DatetimeIndex
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not a DatetimeIndex, attempting to convert")

            # Try multiple strategies to handle the index
            if 'date' in result_df.columns:
                # Strategy 1: Use 'date' column as index
                self.logger.info("Found 'date' column, using it as index")
                result_df['date'] = pd.to_datetime(result_df['date'])
                result_df = result_df.set_index('date')
            elif 'timestamp' in result_df.columns:
                # Strategy 2: Use 'timestamp' column as index
                self.logger.info("Found 'timestamp' column, using it as index")
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                result_df = result_df.set_index('timestamp')
            elif hasattr(result_df.index, 'name') and result_df.index.name in ['date', 'timestamp']:
                # Strategy 3: Current index looks like a date but isn't DatetimeIndex
                self.logger.info(f"Converting existing index '{result_df.index.name}' to DatetimeIndex")
                result_df.index = pd.to_datetime(result_df.index)
            else:
                # Strategy 4: Try to convert the current index directly
                try:
                    self.logger.info("Attempting to convert current index to DatetimeIndex")
                    result_df.index = pd.to_datetime(result_df.index)
                except Exception as e:
                    self.logger.error(f"Failed to convert index to DatetimeIndex: {e}")
                    self.logger.error(f"Index values sample: {result_df.index[:5].tolist()}")
                    raise ValueError(
                        "Cannot convert DataFrame index to DatetimeIndex. "
                        "Please ensure your DataFrame has a datetime index or a 'date'/'timestamp' column."
                    )

        # Verify we now have a DatetimeIndex
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.error("Failed to create DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        self.logger.info(f"Successfully converted to DatetimeIndex: {result_df.index.dtype}")

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

        # Initialize cycle features with proper dtypes
        result_df['days_since_last_halving'] = np.nan
        result_df['days_to_next_halving'] = np.nan
        result_df['halving_cycle_phase'] = np.nan
        result_df['cycle_number'] = 0

        self.logger.info(f"Processing {len(result_df)} data points for halving cycle features")

        # Vectorized approach for better performance
        dates = result_df.index

        # Pre-calculate all cycle information
        days_since_last = np.full(len(dates), np.nan)
        days_to_next = np.full(len(dates), np.nan)
        cycle_phase = np.full(len(dates), np.nan)
        cycle_numbers = np.zeros(len(dates), dtype=int)

        for i, date in enumerate(dates):
            if i % 1000 == 0:
                self.logger.debug(f"Processing date {i}/{len(dates)}: {date}")

            # Find the previous and next halving dates
            previous_halving = None
            next_halving = None
            cycle_number = 0

            for j, halving_date in enumerate(halving_dates):
                if date >= halving_date:
                    previous_halving = halving_date
                    cycle_number = j + 1  # Cycle number starts from 1
                else:
                    next_halving = halving_date
                    break

            # Calculate days since last halving
            if previous_halving is not None:
                days_since_last[i] = (date - previous_halving).days

            # Calculate days to next halving
            if next_halving is not None:
                days_to_next[i] = (next_halving - date).days
            elif previous_halving is not None:
                # If no next halving in our list, estimate based on 4-year cycle
                estimated_next_halving = previous_halving + pd.DateOffset(days=1461)
                days_to_next[i] = (estimated_next_halving - date).days

            # Calculate halving cycle phase (0-1 value representing position in cycle)
            if previous_halving is not None and next_halving is not None:
                cycle_length = (next_halving - previous_halving).days
                days_into_cycle = (date - previous_halving).days
                cycle_phase[i] = days_into_cycle / cycle_length
            elif previous_halving is not None:
                # If we're in the last known cycle, estimate based on 4-year cycle
                cycle_length = 1461  # ~4 years in days
                days_into_cycle = (date - previous_halving).days
                cycle_phase[i] = days_into_cycle / cycle_length

            # Set cycle number
            cycle_numbers[i] = cycle_number

        # Assign all calculated values at once
        result_df['days_since_last_halving'] = days_since_last
        result_df['days_to_next_halving'] = days_to_next
        result_df['halving_cycle_phase'] = cycle_phase
        result_df['cycle_number'] = cycle_numbers

        # Add additional derived features
        self.logger.info("Calculating additional derived features")

        # Log-transformed days since last halving (useful for machine learning)
        result_df['log_days_since_halving'] = np.log1p(result_df['days_since_last_halving'])

        # Sine and cosine transformation for cyclical features (better for neural networks)
        cycle_phase_radians = result_df['halving_cycle_phase'] * 2 * np.pi
        result_df['halving_cycle_sin'] = np.sin(cycle_phase_radians)
        result_df['halving_cycle_cos'] = np.cos(cycle_phase_radians)

        # Relative price change since halving
        # This requires grouping by cycle and calculating percent change from cycle start
        self.logger.info("Calculating price change since halving for each cycle")
        result_df['price_change_since_halving'] = np.nan

        for cycle in result_df['cycle_number'].unique():
            if cycle == 0:  # Skip cycle 0 (before first halving)
                continue

            cycle_mask = result_df['cycle_number'] == cycle
            cycle_data = result_df.loc[cycle_mask]

            if len(cycle_data) == 0:
                continue

            try:
                # Get the first price in this cycle (right after halving)
                cycle_start_price = float(cycle_data['close'].iloc[0])

                if cycle_start_price <= 0:
                    self.logger.warning(f"Invalid starting price for cycle {cycle}: {cycle_start_price}")
                    continue

                # Convert all 'close' values to float within this cycle
                close_values = cycle_data['close'].astype(float)

                # Calculate price change
                price_change = (close_values / cycle_start_price) - 1

                result_df.loc[cycle_mask, 'price_change_since_halving'] = price_change

                self.logger.debug(f"Processed cycle {cycle} with starting price {cycle_start_price}")

            except Exception as e:
                self.logger.error(f"Error calculating price change for cycle {cycle}: {e}")
                self.logger.error(f"Cycle data shape: {cycle_data.shape}")
                if len(cycle_data) > 0:
                    self.logger.error(f"First close price type: {type(cycle_data['close'].iloc[0])}")
                # Continue with other cycles

        # Final data type verification and cleanup
        self.logger.debug("Final data type conversion and validation")
        result_df['days_since_last_halving'] = result_df['days_since_last_halving'].astype('float64')
        result_df['days_to_next_halving'] = result_df['days_to_next_halving'].astype('float64')
        result_df['halving_cycle_phase'] = result_df['halving_cycle_phase'].astype('float64')
        result_df['cycle_number'] = result_df['cycle_number'].astype('int64')

        self.logger.info("Bitcoin halving cycle features calculation completed")
        self.logger.info(f"Output dataframe shape: {result_df.shape}")
        self.logger.info(f"Added features: {[col for col in result_df.columns if col not in processed_data.columns]}")

        return result_df

