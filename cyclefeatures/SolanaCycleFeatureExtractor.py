from utils.config import *
import pandas as pd
import numpy as np


class SolanaCycleFeatureExtractor:
    def __init__(self, market_features):
        self.market_features = market_features
        self.sol_significant_events = sol_significant_events

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