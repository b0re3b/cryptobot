import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from data.db import DatabaseManager
from utils.config import *
from cyclefeatures.BitcoinCycleFeatureExtractor import BitcoinCycleFeatureExtractor
from cyclefeatures.EthereumCycleFeatureExtractor import EthereumCycleFeatureExtractor
from cyclefeatures.SolanaCycleFeatureExtractor import SolanaCycleFeatureExtractor
from cyclefeatures.seasonality import TemporalSeasonalityAnalyzer
from cyclefeatures.MarketPhaseFeatureExtractor import MarketPhaseFeatureExtractor
class CryptoCycles:
    def __init__(self):

        self.db_connection = DatabaseManager()
        self.btcycle = BitcoinCycleFeatureExtractor()
        self.ethcycle = EthereumCycleFeatureExtractor()
        self.solanacycle = SolanaCycleFeatureExtractor()
        self.seasonality = TemporalSeasonalityAnalyzer()
        self.marketplace = MarketPhaseFeatureExtractor()
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
        processed_data = self.db_connection.get_klines(
            symbol=symbol,
            timeframe=timeframe,

        )

        # Cache for future use
        self.cached_processed_data[cache_key] = processed_data

        return processed_data

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
            cycles_data = self.marketplace.identify_bull_bear_cycles(processed_data)
            cycle_column = 'cycle_id'
        elif cycle_type == 'halving' and symbol == 'BTC':
            # Use halving cycles for BTC
            cycles_data = self.btcycle.calculate_btc_halving_cycle_features(processed_data)
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
            cycles_data = self.marketplace.identify_bull_bear_cycles(processed_data)
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
        updated_features = self.marketplace.detect_market_phase(updated_features)

        # Update bull/bear cycle identification
        updated_features = self.marketplace.identify_bull_bear_cycles(updated_features)

        # Update token-specific cycle features
        if symbol_clean == 'BTC':
            updated_features = self.btcycle.calculate_btc_halving_cycle_features(updated_features)
        elif symbol_clean == 'ETH':
            updated_features = self.ethcycle.calculate_eth_event_cycle_features(updated_features)
        elif symbol_clean == 'SOL':
            updated_features = self.solanacycle.calculate_sol_event_cycle_features(updated_features)

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

    def run_full_pipeline(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> dict:

            results = {}

            # 1. Завантаження даних
            data = self.load_processed_data(
                symbol=symbol,
                timeframe=timeframe,

            )
            results['raw_data'] = data.copy()

            # === ВАЖЛИВО: переконайся, що індекс має тип DatetimeIndex ===
            if not isinstance(data.index, pd.DatetimeIndex):
                # Спроба встановити індекс за колонкою open_time
                if 'open_time' in data.columns:
                    data['open_time'] = pd.to_datetime(data['open_time'])  # Конвертуємо в datetime
                    data.set_index('open_time', inplace=True)  # Ставимо індекс
                else:
                    raise ValueError("No 'open_time' column found in data")

            # Перевірка після обробки
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be a DatetimeIndex after conversion.")

            if len(data) == 0:
                raise ValueError(f"No data found for {symbol} in {timeframe} timeframe from {start_date} to {end_date}")

            # 2. Створення циклічних фіч
            features = self.create_cyclical_features(data, symbol)
            results['features'] = features.copy()

            # 3. Аналіз ROI по циклах
            roi_analysis = self.calculate_cycle_roi(features, symbol)
            results['roi_analysis'] = roi_analysis.copy()

            # 4. Виявлення аномалій
            anomalies = self.detect_cycle_anomalies(features, symbol)
            results['anomalies'] = anomalies.copy()

            # 5. Порівняння з історичними циклами
            historical_comparison = self.compare_current_to_historical_cycles(features, symbol)
            results['historical_comparison'] = historical_comparison.copy()

            # 6. Прогнозування точок розвороту
            turning_points = self.predict_cycle_turning_points(features, symbol)
            results['turning_points'] = turning_points.copy()

            # 7. Оновлення фіч з новими даними
            updated_features = self.update_features_with_new_data(features, symbol)
            results['updated_features'] = updated_features.copy()

            return results

    def analyze_multiple_symbols(self, symbols: list, timeframe: str, start_date: str, end_date: str) -> dict:

            results = {}

            for symbol in symbols:
                try:
                    print(f"Processing {symbol}...")
                    results[symbol] = self.run_full_pipeline(symbol, timeframe, start_date, end_date)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    results[symbol] = {"error": str(e)}

            return results


def main():
    from datetime import datetime, timedelta

    # === Налаштування параметрів ===
    symbol = 'BTC'
    timeframe = '1d'
    lookback = '1 year'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Обчислення start_date на основі lookback
    lookback_value, lookback_unit = lookback.split()
    lookback_value = int(lookback_value)

    if 'year' in lookback_unit:
        delta = timedelta(days=365 * lookback_value)
    elif 'month' in lookback_unit:
        delta = timedelta(days=30 * lookback_value)
    elif 'week' in lookback_unit:
        delta = timedelta(weeks=lookback_value)
    elif 'day' in lookback_unit:
        delta = timedelta(days=lookback_value)
    else:
        delta = timedelta(days=365)

    start_date = (datetime.now() - delta).strftime('%Y-%m-%d')

    # === Запуск пайплайну ===
    print(f"Running analysis for {symbol} from {start_date} to {end_date} on {timeframe} timeframe...")

    try:
        crypto_cycles = CryptoCycles()
        results = crypto_cycles.run_full_pipeline(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Виведення результатів
        print("\n=== Analysis Results ===")
        print(f"Processed {len(results['raw_data'])} data points")

        if not results['anomalies'].empty:
            print(f"\nDetected {len(results['anomalies'])} anomalies:")
            print(results['anomalies'][['date', 'anomaly_type', 'description']].head())

        if not results['turning_points'].empty:
            print(f"\nDetected {len(results['turning_points'])} turning points:")
            print(results['turning_points'][['date', 'direction', 'confidence', 'indicators']])

        if 'historical_comparison' in results and 'most_similar_cycle' in results['historical_comparison']:
            print(f"\nMost similar historical cycle: {results['historical_comparison']['most_similar_cycle']}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
