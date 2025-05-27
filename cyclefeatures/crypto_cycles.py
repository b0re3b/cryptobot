import decimal
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from cyclefeatures.BitcoinCycleFeatureExtractor import BitcoinCycleFeatureExtractor
from cyclefeatures.EthereumCycleFeatureExtractor import EthereumCycleFeatureExtractor
from cyclefeatures.MarketPhaseFeatureExtractor import MarketPhaseFeatureExtractor
from cyclefeatures.SolanaCycleFeatureExtractor import SolanaCycleFeatureExtractor
from cyclefeatures.featureextractor import FeatureExtractor
from cyclefeatures.seasonality import TemporalSeasonalityAnalyzer
from data.db import DatabaseManager
from utils.config import *
from utils.logger import CryptoLogger


class CryptoCycles:
    def __init__(self):
        self.db_connection = DatabaseManager()
        self.btcycle = BitcoinCycleFeatureExtractor()
        self.ethcycle = EthereumCycleFeatureExtractor()
        self.solanacycle = SolanaCycleFeatureExtractor()
        self.seasonality = TemporalSeasonalityAnalyzer()
        self.marketplace = MarketPhaseFeatureExtractor()
        self.features = FeatureExtractor()
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
        # Setup logger
        self.logger = CryptoLogger('cyclefeatures')
        self.cached_processed_data = {}

    def _ensure_float_df(self, df):
        """
        Ensure all numeric columns in the dataframe are float type, not Decimal.
        This helps prevent type mismatches in calculations.
        """
        self.logger.debug("Converting dataframe to float type")

        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Check if column contains Decimal values
            if any(isinstance(x, decimal.Decimal) for x in df[col].dropna().head(5)):
                self.logger.debug(f"Converting column {col} from Decimal to float")
                df[col] = df[col].astype(float)

        return df

    def load_processed_data(self, symbol: str, timeframe: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> dict[Any, Any] | None | Any:

        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        self.logger.info(f"Loading data for {symbol} {timeframe} from {start_date} to {end_date}")

        if cache_key in self.cached_processed_data:
            self.logger.debug(f"Using cached data for {cache_key}")
            return self.cached_processed_data[cache_key]

        try:
            # Load pre-processed data from storage manager
            processed_data = self.db_connection.get_klines(
                symbol=symbol,
                timeframe=timeframe,
            )

            if processed_data is None or len(processed_data) == 0:
                self.logger.warning(f"No data found for {symbol} {timeframe}")
                return None

            # Convert any Decimal types to float to avoid type errors
            processed_data = self._ensure_float_df(processed_data)

            # Cache for future use
            self.cached_processed_data[cache_key] = processed_data
            self.logger.debug(f"Loaded {len(processed_data)} rows for {symbol} {timeframe}")

            return processed_data

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
            raise

    def compare_current_to_historical_cycles(self, processed_data: pd.DataFrame,
                                             symbol: str,
                                             cycle_type: str = 'auto',
                                             normalize: bool = True) -> Dict:
        """
        Compare current market cycle to historical cycles.
        """
        self.logger.info(f"Comparing current to historical cycles for {symbol} with cycle_type={cycle_type}")

        # Ensure we have float data to prevent Decimal/float errors
        processed_data = self._ensure_float_df(processed_data)

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

        self.logger.debug(f"Using cycle_type={cycle_type} for {symbol}")

        # Ensure the DataFrame has a datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            self.logger.error("DataFrame index must be a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Identify cycles in the data based on cycle_type
        try:
            if cycle_type == 'bull_bear':
                # Use the bull_bear cycle identification logic
                self.logger.debug("Identifying bull/bear cycles")
                cycles_data = self.marketplace.identify_bull_bear_cycles(processed_data)
                cycle_column = 'cycle_id'
            elif cycle_type == 'halving' and symbol == 'BTC':
                # Use halving cycles for BTC
                self.logger.debug("Calculating BTC halving cycle features")
                cycles_data = self.btcycle.calculate_btc_halving_cycle_features(processed_data)
                cycle_column = 'cycle_number'
            elif cycle_type == 'network_upgrade' and symbol == 'ETH':
                # For ETH, use network upgrades as cycle boundaries
                self.logger.debug("Using ETH network upgrade cycles")
                cycles_data = processed_data.copy()
                # This is just a placeholder - the actual implementation would need
                # to process ETH network upgrade cycles
                cycle_column = 'cycle_id'
            elif cycle_type == 'ecosystem_event' and symbol == 'SOL':
                # For SOL, use ecosystem events as cycle boundaries
                self.logger.debug("Using SOL ecosystem event cycles")
                cycles_data = processed_data.copy()
                # This is just a placeholder - the actual implementation would need
                # to process SOL ecosystem event cycles
                cycle_column = 'cycle_id'
            else:
                # Default to bull/bear cycles for unknown combinations
                self.logger.debug(f"Using default bull/bear cycles for {symbol} with {cycle_type}")
                cycles_data = self.marketplace.identify_bull_bear_cycles(processed_data)
                cycle_column = 'cycle_id'
        except Exception as e:
            self.logger.error(f"Error identifying cycles: {str(e)}")
            self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
            raise

        # Extract the current cycle
        if cycle_column in cycles_data.columns:
            current_cycle_id = cycles_data[cycle_column].iloc[-1]
            self.logger.debug(f"Current cycle ID: {current_cycle_id}")
            current_cycle_data = cycles_data[cycles_data[cycle_column] == current_cycle_id]
        else:
            # If cycle column is not found, assume the last 90 days is the current cycle
            self.logger.warning(f"Cycle column '{cycle_column}' not found, using last 90 days as current cycle")
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
            self.logger.warning("No historical cycles found, grouping by year as fallback")
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
            self.logger.warning("Insufficient data for comparison")
            return {"error": "Insufficient data for comparison", "similarity_scores": {}}

        # Extract price data for comparison
        current_prices = current_cycle_data['close'].values

        # If normalize is True, normalize the current prices
        if normalize:
            self.logger.debug("Normalizing price data")
            current_prices = current_prices / current_prices[0]

        # Compare with each historical cycle
        similarity_scores = {}

        for cycle_id, cycle_data in historical_cycles.items():
            self.logger.debug(f"Comparing with historical cycle {cycle_id}")
            historical_prices = cycle_data['close'].values

            # Skip if not enough data points
            if len(historical_prices) < len(current_prices):
                self.logger.debug(f"Skipping cycle {cycle_id} - insufficient data points")
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
                "similarity": float(similarity),  # Ensure we're using float, not Decimal
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
            self.logger.info(f"Most similar cycle: {most_similar_cycle_id}")
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
                    # Convert to float to avoid Decimal/float issues
                    current_price_end = float(current_prices[-1])
                    historical_price_end = float(
                        most_similar_cycle_data['close'].iloc[best_start_idx + matched_length - 1])
                    historical_price_start = float(most_similar_cycle_data['close'].iloc[best_start_idx])

                    adjustment_factor = current_price_end / (historical_price_end / historical_price_start)
                    predicted_continuation = continuation_segment['close'].values * adjustment_factor
                else:
                    # Convert to float to avoid Decimal/float issues
                    current_price_end = float(current_prices[-1])
                    historical_price_end = float(
                        most_similar_cycle_data['close'].iloc[best_start_idx + matched_length - 1])

                    adjustment_factor = current_price_end / historical_price_end
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
                "confidence_score": float(sorted_scores[most_similar_cycle_id]["similarity"])
                # Ensure we're using float
            }

        self.logger.info(f"Comparison complete. Found {len(sorted_scores)} similar cycles.")
        return comparison_results



    def _convert_to_float(self, value):
        """
        Convert various numeric types (including Decimal) to float.
        """
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        else:
            return 0.0

    def update_features_with_new_data(self, processed_data: pd.DataFrame,
                                      symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """
        Fixed version of update_features_with_new_data with proper error handling
        """
        self.logger.info(f"Updating features for {symbol} with new data")

        if len(processed_data) == 0:
            self.logger.error("No data provided for update")
            raise ValueError("No data provided for update.")

        # Ensure we have a datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            self.logger.error("DataFrame index must be a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Ensure all numeric data is float type
        processed_data = self._ensure_float_df(processed_data)

        # Clean symbol
        symbol_clean = symbol.upper().replace('USDT', '').replace('USD', '')
        self.logger.debug(f"Working with clean symbol: {symbol_clean}")

        # Get the last date in the processed data
        last_date = processed_data.index[-1]
        self.logger.debug(f"Last date in data: {last_date}")

        try:
            # Retrieve the existing cycle features from the database
            self.logger.debug(f"Retrieving existing cycle features for {symbol}")
            existing_features = self.db_connection.get_latest_cycle_features(symbol=symbol, timeframe=timeframe)

            # If no existing features, process all data
            if existing_features is None or len(existing_features) == 0:
                self.logger.info(f"No existing features found for {symbol}, processing all data")
                return self.features.create_cyclical_features(processed_data, symbol)

            # Update features with new data
            self.logger.debug("Updating general market phase detection")
            updated_features = processed_data.copy()

            # Ensure updated_features is using float values
            updated_features = self._ensure_float_df(updated_features)

            try:
                # Update general market phase detection
                updated_features = self.marketplace.detect_market_phase(updated_features)
                self.logger.debug("Market phase detection completed")
            except Exception as e:
                self.logger.error(f"Error in market phase detection: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")

            try:
                # Update bull/bear cycle identification
                updated_features = self.marketplace.identify_bull_bear_cycles(updated_features)
                self.logger.debug("Bull/bear cycle identification completed")
            except Exception as e:
                self.logger.error(f"Error in bull/bear cycle identification: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")

            # Update token-specific cycle features
            try:
                if symbol_clean == 'BTC':
                    self.logger.debug("Calculating BTC halving cycle features")
                    updated_features = self.btcycle.calculate_btc_halving_cycle_features(updated_features)
                elif symbol_clean == 'ETH':
                    self.logger.debug("Calculating ETH event cycle features")
                    updated_features = self.ethcycle.calculate_eth_event_cycle_features(updated_features)
                elif symbol_clean == 'SOL':
                    self.logger.debug("Calculating SOL event cycle features")
                    updated_features = self.solanacycle.calculate_sol_event_cycle_features(updated_features)
            except Exception as e:
                self.logger.error(f"Error in token-specific cycle features calculation: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")

            # Check for new cycle turning points
            try:
                self.logger.debug("Predicting cycle turning points")
                turning_points = self.predict_cycle_turning_points(updated_features, symbol)

                # Save any new turning points to the database
                if turning_points is not None and len(turning_points) > 0:
                    self.logger.info(f"Saving {len(turning_points)} turning points to database")
                    for _, point in turning_points.iterrows():
                        try:
                            # Ensure we're using the right column names and convert to float
                            point_type = str(point.get('direction', 'unknown'))
                            confidence = self._convert_to_float(point.get('confidence', 0.0))
                            description = str(point.get('indicators', 'No description'))

                            self.db_connection.save_predicted_turning_point(
                                symbol=symbol,
                                date=point['date'],
                                point_type=point_type,
                                confidence=confidence,
                                description=description
                            )
                        except Exception as save_error:
                            self.logger.error(f"Error saving turning point: {str(save_error)}")
            except Exception as e:
                self.logger.error(f"Error predicting turning points: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")

            # Check for cycle anomalies
            try:
                self.logger.debug("Detecting cycle anomalies")
                anomalies = self.features.detect_cycle_anomalies(updated_features, symbol)

                # Add anomaly information to features
                if anomalies is not None and len(anomalies) > 0:
                    self.logger.info(f"Found {len(anomalies)} anomalies")
                    for col in anomalies.columns:
                        if col not in updated_features.columns:
                            updated_features[col] = anomalies[col]
            except Exception as e:
                self.logger.error(f"Error detecting cycle anomalies: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")

            # Compare to historical cycles
            try:
                self.logger.debug("Comparing to historical cycles")
                historical_comparison = self.compare_current_to_historical_cycles(
                    self._ensure_float_df(updated_features), symbol)

                # Save the comparison results
                if historical_comparison and 'similarity_scores' in historical_comparison:
                    self.logger.info("Saving historical cycle comparison results")
                    # Iterate through similarity scores
                    for ref_cycle, similarity_data in historical_comparison['similarity_scores'].items():
                        similarity_score = self._convert_to_float(similarity_data.get('similarity', 0.0))

                        self.db_connection.save_cycle_similarity(
                            symbol=symbol,
                            reference_cycle=str(ref_cycle),
                            current_cycle=str(historical_comparison.get('current_cycle_start', 'Unknown')),
                            similarity_score=similarity_score,
                            date=last_date
                        )
            except Exception as e:
                self.logger.error(f"Error comparing to historical cycles: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")

            # Save the updated features to the database
            try:
                self.logger.info(f"Saving updated features for {symbol} to database")
                # Use provided timeframe or get from attrs, default to '1d'
                features_timeframe = timeframe or processed_data.attrs.get('timeframe', '1d')

                self.db_connection.save_cycle_feature(
                    symbol=symbol,
                    timeframe=features_timeframe,
                    features=updated_features,
                    date=last_date
                )
            except Exception as e:
                self.logger.error(f"Error saving cycle features: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")

            self.logger.info(f"Feature update completed for {symbol}")
            self.save_cycle_metrics(
                symbol=symbol,
                metrics=updated_features,
                metric_type='updated_features',
                date=updated_features.index[-1] if isinstance(updated_features.index, pd.DatetimeIndex) else None
            )
            return updated_features

        except Exception as e:
            self.logger.error(f"Error in update_features_with_new_data: {str(e)}")
            self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
            raise

    def predict_cycle_turning_points(self, processed_data: pd.DataFrame,
                                     symbol: str,
                                     cycle_type: str = 'auto',
                                     confidence_interval: float = 0.9) -> pd.DataFrame:
        """
        Fixed version with proper type handling
        """
        self.logger.info(f"Predicting cycle turning points for {symbol} with cycle_type={cycle_type}")

        # Clean symbol format
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

        self.logger.debug(f"Using cycle_type={cycle_type} for {symbol}")

        # Ensure the DataFrame has a datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            self.logger.error("DataFrame index must be a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Create a copy of the processed data and ensure float type for all numeric columns
        df = self._ensure_float_df(processed_data.copy())

        # Get historical cycles comparison for context
        try:
            cycle_comparison = self.compare_current_to_historical_cycles(
                processed_data=df,
                symbol=symbol,
                cycle_type=cycle_type,
                normalize=True
            )
            self.logger.debug(f"Historical cycle comparison completed for {symbol}")
        except Exception as e:
            self.logger.error(f"Error in historical cycle comparison: {str(e)}")
            self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
            cycle_comparison = {}

        # Initialize technical indicators for turning point detection
        self.logger.debug("Calculating technical indicators for turning point detection")

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

        # Avoid division by zero with a small epsilon value
        rs = avg_gain / avg_loss.replace(0, 0.00001)
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

            # Convert all values to float to avoid type conflicts
            rsi_val = self._convert_to_float(row['rsi'])
            close_val = self._convert_to_float(row['close'])
            macd_val = self._convert_to_float(row['macd'])
            macd_signal_val = self._convert_to_float(row['macd_signal'])
            sma_200_val = self._convert_to_float(row['sma_200'])

            # RSI oversold condition
            if rsi_val < 30:
                signals.append('RSI_oversold')
                strength += 1

                # RSI bullish divergence
                if (len(prev_rows) >= 10 and
                        close_val < self._convert_to_float(prev_rows['close'].min()) and
                        rsi_val > self._convert_to_float(prev_rows['rsi'].min())):
                    signals.append('RSI_bullish_divergence')
                    strength += 2

            # MACD bullish crossover
            if (len(prev_rows) >= 2 and
                    macd_val > macd_signal_val and
                    self._convert_to_float(prev_rows['macd'].iloc[-2]) <= self._convert_to_float(
                        prev_rows['macd_signal'].iloc[-2])):
                signals.append('MACD_bullish_cross')
                strength += 1

            # MACD bullish divergence
            if (len(prev_rows) >= 10 and
                    close_val < self._convert_to_float(prev_rows['close'].min()) and
                    macd_val > self._convert_to_float(prev_rows['macd'].min())):
                signals.append('MACD_bullish_divergence')
                strength += 2

            # Moving average support
            if (abs(close_val - sma_200_val) / sma_200_val < 0.03 and
                    close_val < sma_200_val):
                signals.append('MA200_support')
                strength += 1

            # Volume spike
            if 'volume' in row:
                volume_val = self._convert_to_float(row['volume'])
                avg_volume = self._convert_to_float(prev_rows['volume'].mean())
                if volume_val > avg_volume * 2:
                    signals.append('volume_spike')
                    strength += 1

            return signals, strength

        def detect_potential_top(row, prev_rows):
            signals = []
            strength = 0

            # Convert all values to float to avoid type conflicts
            rsi_val = self._convert_to_float(row['rsi'])
            close_val = self._convert_to_float(row['close'])
            macd_val = self._convert_to_float(row['macd'])
            macd_signal_val = self._convert_to_float(row['macd_signal'])
            sma_200_val = self._convert_to_float(row['sma_200'])

            # RSI overbought condition
            if rsi_val > 70:
                signals.append('RSI_overbought')
                strength += 1

                # RSI bearish divergence
                if (len(prev_rows) >= 10 and
                        close_val > self._convert_to_float(prev_rows['close'].max()) and
                        rsi_val < self._convert_to_float(prev_rows['rsi'].max())):
                    signals.append('RSI_bearish_divergence')
                    strength += 2

            # MACD bearish crossover
            if (len(prev_rows) >= 2 and
                    macd_val < macd_signal_val and
                    self._convert_to_float(prev_rows['macd'].iloc[-2]) >= self._convert_to_float(
                        prev_rows['macd_signal'].iloc[-2])):
                signals.append('MACD_bearish_cross')
                strength += 1

            # MACD bearish divergence
            if (len(prev_rows) >= 10 and
                    close_val > self._convert_to_float(prev_rows['close'].max()) and
                    macd_val < self._convert_to_float(prev_rows['macd'].max())):
                signals.append('MACD_bearish_divergence')
                strength += 2

            # Moving average resistance
            if (abs(close_val - sma_200_val) / sma_200_val < 0.03 and
                    close_val > sma_200_val):
                signals.append('MA200_resistance')
                strength += 1

            # Volume spike with price rejection
            if ('volume' in row and 'high' in row and 'low' in row and
                    'open' in row and 'close' in row):
                volume_val = self._convert_to_float(row['volume'])
                avg_volume = self._convert_to_float(prev_rows['volume'].mean())
                high_val = self._convert_to_float(row['high'])
                low_val = self._convert_to_float(row['low'])
                open_val = self._convert_to_float(row['open'])

                if (volume_val > avg_volume * 2 and
                        close_val < open_val and
                        (high_val - max(open_val, close_val)) > (min(open_val, close_val) - low_val)):
                    signals.append('volume_spike_rejection')
                    strength += 1

            return signals, strength

        # Get cycle-specific information based on cycle_type
        if cycle_type == 'halving' and symbol == 'BTC':
            self.logger.debug("Adding BTC halving-specific indicators")
            if 'halving_cycle_phase' in df.columns:
                # Convert to float to prevent type conflicts
                df['days_since_last_halving'] = df['days_since_last_halving'].fillna(0).astype(float)
                df['halving_top_probability'] = (
                    np.exp(-0.5 * ((df['days_since_last_halving'] - 400) / 100) ** 2)
                )

                df['days_to_next_halving'] = df['days_to_next_halving'].fillna(2000).astype(float)
                df['halving_bottom_probability'] = (
                    np.exp(-0.5 * ((df['days_to_next_halving'] - 150) / 50) ** 2)
                )

        # Define lookback window for turning point detection
        lookback_window = 30
        self.logger.debug(f"Using lookback window of {lookback_window} days for turning point detection")

        # Process each data point, skipping the first lookback_window points
        self.logger.debug(f"Analyzing {len(df) - lookback_window} data points for turning points")
        turning_points_count = 0

        for i in range(lookback_window, len(df)):
            row = df.iloc[i]
            prev_rows = df.iloc[i - lookback_window:i]

            # Detect potential bottom
            bottom_signals, bottom_strength = detect_potential_bottom(row, prev_rows)

            # Detect potential top
            top_signals, top_strength = detect_potential_top(row, prev_rows)

            # Adjust strength based on cycle-specific factors
            if cycle_type == 'halving' and symbol == 'BTC' and 'halving_top_probability' in df.columns:
                top_prob = self._convert_to_float(row.get('halving_top_probability', 0))
                bottom_prob = self._convert_to_float(row.get('halving_bottom_probability', 0))
                top_strength = float(top_strength) * (1 + top_prob)
                bottom_strength = float(bottom_strength) * (1 + bottom_prob)

            # Determine if this is a significant turning point
            if bottom_signals and bottom_strength >= 3:
                # Calculate days since last turning point
                if len(turning_points) > 0:
                    days_since_last_tp = (row.name - turning_points['date'].iloc[-1]).days
                else:
                    days_since_last_tp = None

                # Calculate confidence based on strength and other factors
                confidence = min(0.95, float(bottom_strength) / 6)

                # Determine cycle phase
                cycle_phase = self._get_cycle_phase(row)

                # Add to turning points dataframe
                turning_points = pd.concat([turning_points, pd.DataFrame([{
                    'date': row.name,
                    'price': self._convert_to_float(row['close']),
                    'direction': 'bottom',
                    'strength': float(bottom_strength),
                    'confidence': float(confidence),
                    'indicators': ', '.join(bottom_signals),
                    'days_since_last_tp': days_since_last_tp,
                    'cycle_phase': cycle_phase
                }])], ignore_index=True)

                turning_points_count += 1

            elif top_signals and top_strength >= 3:
                # Calculate days since last turning point
                if len(turning_points) > 0:
                    days_since_last_tp = (row.name - turning_points['date'].iloc[-1]).days
                else:
                    days_since_last_tp = None

                # Calculate confidence based on strength and other factors
                confidence = min(0.95, float(top_strength) / 6)

                # Determine cycle phase
                cycle_phase = self._get_cycle_phase(row)

                # Add to turning points dataframe
                turning_points = pd.concat([turning_points, pd.DataFrame([{
                    'date': row.name,
                    'price': self._convert_to_float(row['close']),
                    'direction': 'top',
                    'strength': float(top_strength),
                    'confidence': float(confidence),
                    'indicators': ', '.join(top_signals),
                    'days_since_last_tp': days_since_last_tp,
                    'cycle_phase': cycle_phase
                }])], ignore_index=True)

                turning_points_count += 1

        self.logger.debug(f"Detected {turning_points_count} preliminary turning points")

        # Filter turning points by the confidence interval
        turning_points = turning_points[turning_points['confidence'] >= confidence_interval]
        self.logger.debug(f"After confidence filtering: {len(turning_points)} turning points remain")

        # Use historical cycle comparison to predict future turning points
        if cycle_comparison and 'potential_continuation' in cycle_comparison:
            turning_points = self._add_projected_turning_points(turning_points, cycle_comparison, df)

        # Add future event-based turning points
        turning_points = self._add_future_events(turning_points, symbol, cycle_type, df)

        # Sort by date again after adding events
        turning_points = turning_points.sort_values('date')

        self.logger.info(f"Completed turning point prediction for {symbol}. Found {len(turning_points)} points.")
        return turning_points

    def _get_cycle_phase(self, row):
        """Helper method to determine cycle phase"""
        if 'market_phase' in row:
            return str(row['market_phase'])
        elif 'halving_cycle_phase' in row:
            phase_val = self._convert_to_float(row['halving_cycle_phase'])
            return f"Halving cycle: {phase_val:.2f}"
        else:
            return "Unknown"

    def _add_projected_turning_points(self, turning_points, cycle_comparison, df):
        """Helper method to add projected turning points based on historical patterns"""
        self.logger.debug("Using historical pattern to project future turning points")
        potential_prices = cycle_comparison['potential_continuation']['prices']

        if len(potential_prices) > 10:
            # Calculate the last known date and project future dates
            last_date = df.index[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, len(potential_prices) + 1)]

            # Create a DataFrame with projected prices
            projected_df = pd.DataFrame({
                'close': [self._convert_to_float(p) for p in potential_prices],
                'date': future_dates
            })
            projected_df.set_index('date', inplace=True)

            # Calculate basic metrics for the projected data
            projected_df['price_change'] = projected_df['close'].pct_change()

            # Detect potential future turning points
            lookback = min(10, len(projected_df))
            projected_tops = 0
            projected_bottoms = 0

            for i in range(lookback, len(projected_df)):
                curr_price = self._convert_to_float(projected_df['close'].iloc[i])
                prev_prices = projected_df['close'].iloc[i - lookback:i].apply(self._convert_to_float)

                # Simple peak detection
                if all(curr_price > prev_prices):
                    # Possible future top
                    confidence = min(0.8, self._convert_to_float(
                        cycle_comparison['potential_continuation'].get('confidence_score', 0.5)))

                    turning_points = pd.concat([turning_points, pd.DataFrame([{
                        'date': projected_df.index[i],
                        'price': curr_price,
                        'direction': 'projected_top',
                        'strength': 3.0,
                        'confidence': float(confidence),
                        'indicators': 'Historical pattern projection',
                        'days_since_last_tp': (projected_df.index[i] - turning_points['date'].iloc[-1]).days if len(
                            turning_points) > 0 else None,
                        'cycle_phase': f"Projected ({cycle_comparison.get('most_similar_cycle', 'Unknown')})"
                    }])], ignore_index=True)
                    projected_tops += 1

                # Simple valley detection
                elif all(curr_price < prev_prices):
                    # Possible future bottom
                    confidence = min(0.8, self._convert_to_float(
                        cycle_comparison['potential_continuation'].get('confidence_score', 0.5)))

                    turning_points = pd.concat([turning_points, pd.DataFrame([{
                        'date': projected_df.index[i],
                        'price': curr_price,
                        'direction': 'projected_bottom',
                        'strength': 3.0,
                        'confidence': float(confidence),
                        'indicators': 'Historical pattern projection',
                        'days_since_last_tp': (projected_df.index[i] - turning_points['date'].iloc[-1]).days if len(
                            turning_points) > 0 else None,
                        'cycle_phase': f"Projected ({cycle_comparison.get('most_similar_cycle', 'Unknown')})"
                    }])], ignore_index=True)
                    projected_bottoms += 1

            self.logger.debug(f"Added {projected_tops} projected tops and {projected_bottoms} projected bottoms")

        return turning_points

    def run_full_pipeline(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> dict:

        self.logger.info(f"Running full pipeline for {symbol} from {start_date} to {end_date} on {timeframe} timeframe")
        results = {}

        try:
            # 1. Load data
            self.logger.debug(f"Loading data for {symbol}")
            data = self.load_processed_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if data is None:
                self.logger.error(f"No data found for {symbol}")
                raise ValueError(f"No data found for {symbol}")

            results['raw_data'] = data.copy()

            # Ensure DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.debug("Converting index to DatetimeIndex")
                # Try to set index from open_time column
                if 'open_time' in data.columns:
                    data['open_time'] = pd.to_datetime(data['open_time'])
                    data.set_index('open_time', inplace=True)
                else:
                    self.logger.error("No 'open_time' column found in data")
                    raise ValueError("No 'open_time' column found in data")

            # Verify after processing
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.error("DataFrame index must be a DatetimeIndex after conversion")
                raise ValueError("DataFrame index must be a DatetimeIndex after conversion")

            if len(data) == 0:
                self.logger.error(
                    f"No data found for {symbol} in {timeframe} timeframe from {start_date} to {end_date}")
                raise ValueError(f"No data found for {symbol} in {timeframe} timeframe from {start_date} to {end_date}")

            # Ensure all data is float type to avoid Decimal/float issues
            data = self._ensure_float_df(data)

            # 2. Create cyclical features
            self.logger.debug("Creating cyclical features")
            features = self.features.create_cyclical_features(data, symbol)
            results['features'] = features.copy()

            # 3. Analyze ROI by cycles
            self.logger.debug("Calculating cycle ROI")
            try:
                roi_analysis = self.features.calculate_cycle_roi(features, symbol)
                results['roi_analysis'] = roi_analysis.copy()
            except Exception as e:
                self.logger.error(f"Error calculating cycle ROI: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                results['roi_analysis'] = pd.DataFrame()

            # 4. Detect anomalies
            self.logger.debug("Detecting cycle anomalies")
            try:
                anomalies = self.features.detect_cycle_anomalies(features, symbol)
                results['anomalies'] = anomalies.copy() if anomalies is not None else pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error detecting cycle anomalies: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                results['anomalies'] = pd.DataFrame()

            # 5. Compare with historical cycles
            self.logger.debug("Comparing with historical cycles")
            try:
                historical_comparison = self.compare_current_to_historical_cycles(features, symbol)
                results['historical_comparison'] = historical_comparison.copy() if isinstance(historical_comparison,
                                                                                              pd.DataFrame) else historical_comparison
            except Exception as e:
                self.logger.error(f"Error comparing with historical cycles: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                results['historical_comparison'] = {}

            # 6. Predict turning points
            self.logger.debug("Predicting turning points")
            try:
                turning_points = self.predict_cycle_turning_points(features, symbol)
                results['turning_points'] = turning_points.copy()
            except Exception as e:
                self.logger.error(f"Error predicting turning points: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                results['turning_points'] = pd.DataFrame()

            # 7. Update features with new data
            self.logger.debug("Updating features with new data")
            try:
                updated_features = self.update_features_with_new_data(features, symbol)
                results['updated_features'] = updated_features.copy()
            except Exception as e:
                self.logger.error(f"Error updating features: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                results['updated_features'] = features.copy()

            self.logger.info(f"Full pipeline completed for {symbol}")
            return results

        except Exception as e:
            self.logger.error(f"Error in run_full_pipeline: {str(e)}")
            self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
            raise

    def analyze_multiple_symbols(self, symbols: list, timeframe: str, start_date: str, end_date: str) -> dict:

        self.logger.info(f"Analyzing multiple symbols: {symbols} from {start_date} to {end_date}")
        results = {}

        for symbol in symbols:
            try:
                self.logger.info(f"Processing {symbol}...")
                results[symbol] = self.run_full_pipeline(symbol, timeframe, start_date, end_date)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                results[symbol] = {"error": str(e)}

        self.logger.info(f"Completed analysis of {len(symbols)} symbols")
        return results

    def save_cycle_metrics(self, processed_data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        self.logger.info(f"Saving cycle metrics for {symbol} on {timeframe} timeframe")

        try:
            # Validate input data
            if processed_data is None or len(processed_data) == 0:
                self.logger.error("No data to save")
                return False

            # Ensure we're working with float data to prevent Decimal/float type issues
            processed_data = self._ensure_float_df(processed_data)

            # Get the latest data point
            latest_data = processed_data.iloc[-1]

            # Determine timestamp from index if it's a DatetimeIndex, otherwise use current time
            timestamp = latest_data.name if isinstance(processed_data.index, pd.DatetimeIndex) else datetime.now()

            # Clean symbol format
            symbol_clean = symbol.upper().replace('USDT', '').replace('USD', '')

            # Create a metrics dictionary with default values
            metrics = {
                # Default values for all cryptos
                'symbol': symbol,
                'timestamp': timestamp,
                'timeframe': timeframe,
                'weekly_cycle_position': 0.0,
                'monthly_seasonality_factor': 0.0,
                'market_phase': 'unknown',
                'optimal_cycle_length': 0,
                'btc_correlation': 0.0,
                'eth_correlation': 0.0,
                'sol_correlation': 0.0,
                'volatility_metric': 0.0,
                'is_anomaly': False,

                # Asset-specific metrics (with defaults)
                'days_since_last_halving': 0,
                'days_to_next_halving': 0,
                'halving_cycle_phase': 0.0,
                'days_since_last_eth_upgrade': 0,
                'days_to_next_eth_upgrade': 0,
                'eth_upgrade_cycle_phase': 0.0,
                'days_since_last_sol_event': 0,
                'sol_network_stability_score': 0.0,
            }

            # Extract general metrics that should be available for all symbols
            for metric in ['weekly_cycle_position', 'monthly_seasonality_factor', 'market_phase',
                           'optimal_cycle_length', 'btc_correlation', 'eth_correlation',
                           'sol_correlation', 'volatility', 'is_anomaly']:
                if metric in processed_data.columns:
                    # Type conversion based on expected type
                    if metric in ['optimal_cycle_length']:
                        metrics[metric] = int(latest_data.get(metric, metrics[metric]))
                    elif metric in ['is_anomaly']:
                        metrics[metric] = bool(latest_data.get(metric, metrics[metric]))
                    elif metric in ['market_phase']:
                        metrics[metric] = str(latest_data.get(metric, metrics[metric]))
                    else:
                        metrics[metric] = float(latest_data.get(metric, metrics[metric]))

            # For volatility, the column name might be different
            if 'volatility' in processed_data.columns:
                metrics['volatility_metric'] = float(latest_data.get('volatility', 0.0))

            # Extract BTC-specific metrics
            if symbol_clean == 'BTC':
                for metric in ['days_since_last_halving', 'days_to_next_halving', 'halving_cycle_phase']:
                    if metric in processed_data.columns:
                        if metric in ['days_since_last_halving', 'days_to_next_halving']:
                            metrics[metric] = int(latest_data.get(metric, metrics[metric]))
                        else:
                            metrics[metric] = float(latest_data.get(metric, metrics[metric]))

            # Extract ETH-specific metrics
            elif symbol_clean == 'ETH':
                for metric in ['days_since_last_eth_upgrade', 'days_to_next_eth_upgrade', 'eth_upgrade_cycle_phase']:
                    if metric in processed_data.columns:
                        if metric in ['days_since_last_eth_upgrade', 'days_to_next_eth_upgrade']:
                            metrics[metric] = int(latest_data.get(metric, metrics[metric]))
                        else:
                            metrics[metric] = float(latest_data.get(metric, metrics[metric]))

            # Extract SOL-specific metrics
            elif symbol_clean == 'SOL':
                for metric in ['days_since_last_sol_event', 'sol_network_stability_score']:
                    if metric in processed_data.columns:
                        if metric == 'days_since_last_sol_event':
                            metrics[metric] = int(latest_data.get(metric, metrics[metric]))
                        else:
                            metrics[metric] = float(latest_data.get(metric, metrics[metric]))

            # Log the metrics being saved
            self.logger.debug(f"Saving metrics for {symbol}: {metrics}")

            # Save cycle features to the database using the provided method
            cycle_id = self.db_connection.save_cycle_feature(
                symbol=symbol,
                timestamp=timestamp,
                timeframe=timeframe,
                days_since_last_halving=metrics['days_since_last_halving'],
                days_to_next_halving=metrics['days_to_next_halving'],
                halving_cycle_phase=metrics['halving_cycle_phase'],
                days_since_last_eth_upgrade=metrics['days_since_last_eth_upgrade'],
                days_to_next_eth_upgrade=metrics['days_to_next_eth_upgrade'],
                eth_upgrade_cycle_phase=metrics['eth_upgrade_cycle_phase'],
                days_since_last_sol_event=metrics['days_since_last_sol_event'],
                sol_network_stability_score=metrics['sol_network_stability_score'],
                weekly_cycle_position=metrics['weekly_cycle_position'],
                monthly_seasonality_factor=metrics['monthly_seasonality_factor'],
                market_phase=metrics['market_phase'],
                optimal_cycle_length=metrics['optimal_cycle_length'],
                btc_correlation=metrics['btc_correlation'],
                eth_correlation=metrics['eth_correlation'],
                sol_correlation=metrics['sol_correlation'],
                volatility_metric=metrics['volatility_metric'],
                is_anomaly=metrics['is_anomaly']
            )

            self.logger.info(f"Saved cycle features with ID: {cycle_id}")

            # Save cycle similarity data if available
            try:
                # Compare current cycle with historical ones
                historical_comparison = self.compare_current_to_historical_cycles(
                    processed_data=processed_data,
                    symbol=symbol,
                    normalize=True
                )

                if historical_comparison and 'similarity_scores' in historical_comparison:
                    self.logger.debug(f"Saving {len(historical_comparison['similarity_scores'])} similarity scores")

                    for cycle_id, similarity_data in historical_comparison['similarity_scores'].items():
                        # Better error handling for invalid cycle IDs
                        ref_cycle_id = int(cycle_id) if isinstance(cycle_id, str) and cycle_id.isdigit() else 0

                        current_cycle = historical_comparison.get('current_cycle', '0')
                        compared_cycle_id = int(current_cycle) if isinstance(current_cycle,
                                                                             str) and current_cycle.isdigit() else 0

                        similarity_score = float(similarity_data.get('similarity', 0.0))

                        # Save using the provided method
                        similarity_id = self.db_connection.save_cycle_similarity(
                            symbol=symbol,
                            reference_cycle_id=ref_cycle_id,
                            compared_cycle_id=compared_cycle_id,
                            similarity_score=similarity_score,
                            normalized=True
                        )
                        self.logger.debug(f"Saved similarity record with ID: {similarity_id}")

            except Exception as e:
                self.logger.error(f"Error saving cycle similarity data: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                # Continue execution despite similarity saving error

            # Save turning points predictions if available
            try:
                turning_points = self.predict_cycle_turning_points(processed_data, symbol)

                if turning_points is not None and len(turning_points) > 0:
                    # Only save future turning points
                    current_date = datetime.now()
                    future_points = turning_points[pd.to_datetime(turning_points['date']) > current_date]

                    self.logger.debug(f"Saving {len(future_points)} future turning points")

                    for _, point in future_points.iterrows():
                        point_type = point.get('direction', 'unknown')
                        confidence = float(point.get('confidence', 0.0))
                        predicted_date = point['date']

                        # Handle None or NaN price values
                        predicted_price = None
                        if 'price' in point and pd.notna(point['price']):
                            predicted_price = float(point['price'])

                        # Save using the provided method
                        tp_id = self.db_connection.save_predicted_turning_point(
                            symbol=symbol,
                            prediction_date=timestamp,
                            predicted_point_date=predicted_date,
                            point_type=point_type,
                            confidence=confidence,
                            price_prediction=predicted_price
                        )
                        self.logger.debug(f"Saved turning point prediction with ID: {tp_id}")

            except Exception as e:
                self.logger.error(f"Error saving turning point predictions: {str(e)}")
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
                # Continue execution despite turning point saving error

            return True

        except Exception as e:
            self.logger.error(f"Error saving cycle metrics: {str(e)}")
            self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
            return False

    @staticmethod
    def prepare_cycle_ml_features(processed_data: pd.DataFrame, symbol: str) -> pd.DataFrame:

        logger = CryptoLogger('FeaturePreparation')

        try:
            # Create a copy of the input data
            features_df = processed_data.copy()

            # Clean symbol
            clean_symbol = symbol.upper().replace('USDT', '').replace('USD', '')
            logger.info(f"Preparing ML features for {clean_symbol}")

            # 1. Add token-specific cycle features
            if clean_symbol == 'BTC':
                btc_extractor = BitcoinCycleFeatureExtractor()
                features_df = btc_extractor.calculate_btc_halving_cycle_features(features_df)
            elif clean_symbol == 'ETH':
                eth_extractor = EthereumCycleFeatureExtractor()
                features_df = eth_extractor.calculate_eth_event_cycle_features(features_df)
            elif clean_symbol == 'SOL':
                sol_extractor = SolanaCycleFeatureExtractor()
                features_df = sol_extractor.calculate_sol_event_cycle_features(features_df)

            # 2. Add market phase features
            market_phase_extractor = MarketPhaseFeatureExtractor()
            features_df = market_phase_extractor.detect_market_phase(features_df)
            features_df = market_phase_extractor.identify_bull_bear_cycles(features_df)

            # 3. Add seasonality features
            seasonality_analyzer = TemporalSeasonalityAnalyzer()

            # Weekly features
            weekly_stats = seasonality_analyzer.analyze_weekly_cycle(features_df)
            if 'day_of_week' not in features_df.columns:
                features_df['day_of_week'] = features_df.index.dayofweek
            features_df['day_of_week_sin'] = np.sin(features_df['day_of_week'] * (2 * np.pi / 7))
            features_df['day_of_week_cos'] = np.cos(features_df['day_of_week'] * (2 * np.pi / 7))

            # Monthly features
            monthly_stats = seasonality_analyzer.analyze_monthly_seasonality(features_df)
            features_df['month'] = features_df.index.month
            features_df['month_sin'] = np.sin(features_df['month'] * (2 * np.pi / 12))
            features_df['month_cos'] = np.cos(features_df['month'] * (2 * np.pi / 12))

            # 4. Add technical indicators
            features_df['returns'] = features_df['close'].pct_change()
            features_df['volatility'] = features_df['returns'].rolling(14).std()

            # RSI
            delta = features_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, 0.00001)
            features_df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            features_df['ema12'] = features_df['close'].ewm(span=12, adjust=False).mean()
            features_df['ema26'] = features_df['close'].ewm(span=26, adjust=False).mean()
            features_df['macd'] = features_df['ema12'] - features_df['ema26']
            features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()

            # 5. Create lagged features for time series
            for lag in [1, 2, 3, 5, 7, 14]:
                features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
                features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
                features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)

            # 6. Drop unnecessary columns and handle missing values
            features_df = features_df.dropna()
            features_df = features_df.drop(columns=['day_of_week', 'month'], errors='ignore')

            logger.info(f"Successfully prepared {len(features_df.columns)} features for ML model")
            return features_df

        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            raise
def main():
        from datetime import datetime, timedelta
        import logging

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("crypto_cycles.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger("CryptoCycles")
        logger.info("Starting CryptoCycles analysis")

        # === Parameters setup ===
        symbol = 'BTC'
        timeframe = '1h'
        lookback = '1 year'
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Calculate start_date based on lookback
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

        # === Run pipeline ===
        logger.info(f"Running analysis for {symbol} from {start_date} to {end_date} on {timeframe} timeframe...")

        try:
            crypto_cycles = CryptoCycles()
            results = crypto_cycles.run_full_pipeline(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # Print results
            logger.info("\n=== Analysis Results ===")
            logger.info(f"Processed {len(results['raw_data'])} data points")

            if 'anomalies' in results and not results['anomalies'].empty:
                logger.info(f"\nDetected {len(results['anomalies'])} anomalies:")
                logger.info(results['anomalies'][['date', 'anomaly_type', 'description']].head())

            if 'turning_points' in results and not results['turning_points'].empty:
                logger.info(f"\nDetected {len(results['turning_points'])} turning points:")
                logger.info(results['turning_points'][['date', 'direction', 'confidence', 'indicators']])

            if 'historical_comparison' in results and 'most_similar_cycle' in results['historical_comparison']:
                logger.info(
                    f"\nMost similar historical cycle: {results['historical_comparison']['most_similar_cycle']}")

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            logger.debug(f"Exception traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()