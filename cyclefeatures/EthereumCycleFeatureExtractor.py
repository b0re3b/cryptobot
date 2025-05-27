import numpy as np
import pandas as pd
import decimal
from utils.config import *
from utils.logger import CryptoLogger


class EthereumCycleFeatureExtractor:
    def __init__(self):
        self.eth_significant_events = eth_significant_events
        self.logger = CryptoLogger('EthereumCycleFeatureExtractor')

    def calculate_eth_event_cycle_features(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
            Обчислює різноманітні ознаки (features) на основі циклів важливих подій Ethereum (Ethereum upgrade events)
            для заданого DataFrame з часовою індексацією.

            Метод додає до вхідного DataFrame кілька нових колонок, які характеризують:
            - Кількість днів від останнього оновлення Ethereum (days_since_last_upgrade)
            - Кількість днів до наступного відомого оновлення (days_to_next_known_upgrade)
            - Поточну фазу циклу оновлень (upgrade_cycle_phase), значення від 0 до 1
            - Індикатор фази ETH 2.0 (eth2_phase_indicator), цілочисельне значення від 0 до 4
            - Індикатор прогресу переходу на Proof of Stake (pos_transition_indicator), значення від 0 до 1
            - Логарифмічне перетворення днів від останнього оновлення (log_days_since_upgrade)
            - Синус та косинус циклічного показника оновлення (upgrade_cycle_sin, upgrade_cycle_cos)
            - Важливість оновлення (upgrade_importance) на основі історичного впливу на ціну

            Args:
                processed_data (pd.DataFrame): Вхідний DataFrame з часовою індексацією (DatetimeIndex),
                    що містить дані для аналізу. Індекс має бути типу DatetimeIndex.

            Returns:
                pd.DataFrame: Копія вхідного DataFrame з додатковими колонками, що містять розраховані
                    характеристики циклів оновлень Ethereum.

            Викидає:
                ValueError: Якщо індекс DataFrame не є DatetimeIndex.

            Особливості:
                - Підтримує фіксовані дати основних фаз ETH 2.0 та ключових оновлень.
                - Логування виконання та помилок через self.logger.
                - Обробляє наявність майбутніх оновлень (next_known_upgrade) – зараз в методі закладено
                  заглушку, яку можна замінити на реальну логіку.
            """
        self.logger.info("Starting calculation of Ethereum cycle features")

        # Create a copy of the input DataFrame to avoid modifying the original
        result_df = processed_data.copy()
        self.logger.debug(f"Input DataFrame shape: {result_df.shape}")

        # Ensure the DataFrame has a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.error("DataFrame index is not a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Convert Ethereum events to datetime objects and sort them
        eth_events = sorted(self.eth_significant_events, key=lambda x: pd.Timestamp(x["date"]))
        eth_event_dates = [pd.Timestamp(event["date"]) for event in eth_events]
        self.logger.info(f"Found {len(eth_events)} Ethereum significant events")

        # Add known future Ethereum upgrades if they exist
        next_known_upgrade = None
        # This logic would be replaced with actual lookup logic in a real implementation

        # Initialize features
        result_df['days_since_last_upgrade'] = None
        result_df['days_to_next_known_upgrade'] = None
        result_df['upgrade_cycle_phase'] = None
        result_df['eth2_phase_indicator'] = 0  # ETH 2.0 phase indicator (0-4)
        result_df['pos_transition_indicator'] = 0  # Proof of Stake transition progress (0-1)
        self.logger.debug("Initialized feature columns")

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
        self.logger.info(f"Defined {len(eth2_phases)} ETH 2.0 phases")

        # Calculate features for each date in the DataFrame
        total_dates = len(result_df.index)
        self.logger.info(f"Processing {total_dates} dates")

        for idx, date in enumerate(result_df.index):
            if idx % 1000 == 0:  # Log progress every 1000 entries
                self.logger.debug(f"Processing date {idx + 1}/{total_dates}: {date}")

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
                try:
                    # Convert to float explicitly to avoid decimal/float incompatibility
                    cycle_length = float((next_upgrade - previous_upgrade).days)
                    days_into_cycle = float((date - previous_upgrade).days)

                    # Check for division by zero
                    if cycle_length > 0:
                        cycle_phase = days_into_cycle / cycle_length
                    else:
                        self.logger.warning(f"Zero cycle length detected for date {date}")
                        cycle_phase = 0.0

                    result_df.at[date, 'upgrade_cycle_phase'] = cycle_phase
                except (TypeError, ValueError, decimal.InvalidOperation) as e:
                    self.logger.error(f"Error calculating cycle phase for date {date}: {str(e)}")
                    self.logger.debug(
                        f"cycle_length type: {type(cycle_length)}, days_into_cycle type: {type(days_into_cycle)}")
                    result_df.at[date, 'upgrade_cycle_phase'] = np.nan
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
                try:
                    # Convert to float explicitly to avoid decimal/float incompatibility
                    total_transition_days = float((merge_date - beacon_chain_date).days)
                    days_since_beacon = float((date - beacon_chain_date).days)

                    # Check for division by zero
                    if total_transition_days > 0:
                        transition_progress = days_since_beacon / total_transition_days
                    else:
                        self.logger.warning(f"Zero transition days detected for date {date}")
                        transition_progress = 0.0

                    result_df.at[date, 'pos_transition_indicator'] = transition_progress
                except (TypeError, ValueError, decimal.InvalidOperation) as e:
                    self.logger.error(f"Error calculating PoS transition for date {date}: {str(e)}")
                    self.logger.debug(
                        f"total_transition_days type: {type(total_transition_days)}, days_since_beacon type: {type(days_since_beacon)}")
                    result_df.at[date, 'pos_transition_indicator'] = 0.0

        # Convert features to appropriate data types
        self.logger.debug("Converting features to appropriate data types")
        result_df['days_since_last_upgrade'] = result_df['days_since_last_upgrade'].astype('float64')
        result_df['days_to_next_known_upgrade'] = result_df['days_to_next_known_upgrade'].astype('float64')
        result_df['upgrade_cycle_phase'] = result_df['upgrade_cycle_phase'].astype('float64')
        result_df['eth2_phase_indicator'] = result_df['eth2_phase_indicator'].astype('int64')
        result_df['pos_transition_indicator'] = result_df['pos_transition_indicator'].astype('float64')

        # Add additional derived features
        self.logger.info("Calculating derived features")

        # Log-transformed days since last upgrade (useful for machine learning)
        result_df['log_days_since_upgrade'] = np.log1p(result_df['days_since_last_upgrade'])

        # Sine and cosine transformation for cyclical features (better for neural networks)
        try:
            cycle_phase = result_df['upgrade_cycle_phase'] * 2 * np.pi
            result_df['upgrade_cycle_sin'] = np.sin(cycle_phase)
            result_df['upgrade_cycle_cos'] = np.cos(cycle_phase)
            self.logger.debug("Calculated sine and cosine cyclical features")
        except Exception as e:
            self.logger.error(f"Error calculating cyclical features: {str(e)}")
            # Set default values if calculation fails
            result_df['upgrade_cycle_sin'] = 0.0
            result_df['upgrade_cycle_cos'] = 0.0

        # Add upgrade importance weight based on historical price impact
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
        self.logger.debug(f"Defined importance values for {len(upgrade_importance)} upgrades")

        # Apply importance values based on the most recent upgrade
        for idx, date in enumerate(result_df.index):
            if idx % 1000 == 0:  # Log progress every 1000 entries
                self.logger.debug(f"Processing importance for date {idx + 1}/{total_dates}")

            for event in eth_events:
                event_date = pd.Timestamp(event["date"])
                if date >= event_date:
                    importance = upgrade_importance.get(event["name"], 0.3)  # Default to 0.3
                    result_df.at[date, 'upgrade_importance'] = importance

        self.logger.info(f"Finished calculating Ethereum cycle features. Result DataFrame shape: {result_df.shape}")
        return result_df