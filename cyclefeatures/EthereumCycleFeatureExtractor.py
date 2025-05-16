import numpy as np
import pandas as pd
from utils.config import *

class EthereumCycleFeatureExtractor:
    def __init__(self):
        self.eth_significant_events = eth_significant_events
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