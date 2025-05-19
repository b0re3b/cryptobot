import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from data.db import DatabaseManager
from timeseriesmodels import TimeSeriesAnalyzer, ARIMAModeler
from timeseriesmodels.ModelEvaluator import ModelEvaluator
from timeseriesmodels.TimeSeriesTransformer import TimeSeriesTransformer
from timeseriesmodels.Forecaster import Forecaster
from utils.logger import CryptoLogger


class TimeSeriesModels:

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.models = {}  # Dictionary for storing trained models
        self.transformations = {}  # Dictionary for storing transformation parameters
        self.modeler = ARIMAModeler()
        self.transformer = TimeSeriesTransformer()  # Fixed attribute name
        # Logging setup
        self.forecaster = Forecaster()  # Fixed attribute name (capitalization)
        self.logger = CryptoLogger('TimeSeriesModels')
        self.analyzer = TimeSeriesAnalyzer()
        self.evaluator = ModelEvaluator()

        self.logger.info("TimeSeriesModels initialized")

    def load_crypto_data(self, db_manager: Any,
                         symbol: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         timeframe: str = '1d') -> pd.DataFrame:

        try:
            symbol = symbol.upper()  # Для стандартизації працюємо з верхнім регістром
            self.logger.info(f"Loading {symbol} data with interval {timeframe} from database")

            # Зберігаємо посилання на db_manager
            self.db_manager = db_manager

            if not hasattr(self, 'db_manager') or self.db_manager is None:
                self.logger.error("db_manager not initialized in TimeSeriesModels class")
                raise ValueError("db_manager not available. Please initialize db_manager.")

            # Мапування криптовалют до методів отримання даних
            crypto_methods = {
                'BTC': self.db_manager.get_btc_arima_data,
                'ETH': self.db_manager.get_eth_arima_data,
                'SOL': self.db_manager.get_sol_arima_data
            }

            # Перевіряємо, чи підтримується вказана криптовалюта
            if symbol not in crypto_methods:
                self.logger.error(f"Unsupported cryptocurrency: {symbol}")
                return pd.DataFrame()

            # Отримуємо дані через відповідний метод
            klines_data = crypto_methods[symbol](timeframe=timeframe)

            # Перевіряємо чи є дані
            if klines_data is None or (isinstance(klines_data, pd.DataFrame) and klines_data.empty):
                self.logger.warning(f"No data found for {symbol} with interval {timeframe}")
                return pd.DataFrame()

            # Конвертуємо в DataFrame якщо потрібно
            if not isinstance(klines_data, pd.DataFrame):
                klines_data = pd.DataFrame(klines_data)
                self.logger.info("Converted data to DataFrame")

            # Обробка часового індексу
            if not isinstance(klines_data.index, pd.DatetimeIndex):
                # Шукаємо колонку з часом
                time_cols = [col for col in klines_data.columns if any(
                    time_str in str(col).lower() for time_str in ['time', 'date', 'timestamp'])]

                if time_cols:
                    try:
                        klines_data[time_cols[0]] = pd.to_datetime(klines_data[time_cols[0]])
                        klines_data = klines_data.set_index(time_cols[0])
                        self.logger.info(f"Set index using column: {time_cols[0]}")
                    except Exception as e:
                        self.logger.warning(f"Failed to convert {time_cols[0]} to datetime index: {str(e)}")
                else:
                    self.logger.warning("No time column found in data. Using default index.")

            # Сортування та фільтрація за датами
            if isinstance(klines_data.index, pd.DatetimeIndex):
                klines_data = klines_data.sort_index()
                if start_date:
                    klines_data = klines_data[klines_data.index >= start_date]
                if end_date:
                    klines_data = klines_data[klines_data.index <= end_date]

            # Логування успішного завантаження
            if not klines_data.empty and isinstance(klines_data.index, pd.DatetimeIndex):
                self.logger.info(f"Loaded {len(klines_data)} records for {symbol} "
                                 f"from {klines_data.index.min()} to {klines_data.index.max()}")
            else:
                self.logger.info(f"Loaded {len(klines_data)} records for {symbol}")

            return klines_data

        except Exception as e:
            self.logger.error(f"Error loading crypto data for {symbol}: {str(e)}")
            self.logger.exception("Stack trace:")
            return pd.DataFrame()

    def save_forecast_to_db(self, db_manager: Any, symbol: str,
                            forecast_data: pd.Series, model_key: str) -> bool:

        try:
            self.logger.info(f"Saving forecast for {symbol} using model {model_key}")

            if forecast_data is None or len(forecast_data) == 0:
                self.logger.error("No forecast data provided")
                return False

            # Check if forecast_data is pd.Series
            if not isinstance(forecast_data, pd.Series):
                try:
                    forecast_data = pd.Series(forecast_data)
                    self.logger.warning("Converted forecast data to pandas Series")
                except Exception as convert_error:
                    self.logger.error(f"Could not convert forecast data to pandas Series: {str(convert_error)}")
                    return False

            # Use the provided db_manager or the one saved in the class
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return False

            # Check if the model exists in the database
            model_info = manager.get_model_by_key(model_key)

            if model_info is None:
                self.logger.error(f"Model with key {model_key} not found in database")
                return False

            # Convert forecast data to a format for saving
            forecast_dict = {
                "model_key": model_key,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "forecast_data": {
                    timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp): value
                    for timestamp, value in forecast_data.items()
                },
                "forecast_horizon": len(forecast_data),
                "forecast_start": forecast_data.index[0].isoformat() if isinstance(forecast_data.index[0], datetime)
                else str(forecast_data.index[0]),
                "forecast_end": forecast_data.index[-1].isoformat() if isinstance(forecast_data.index[-1], datetime)
                else str(forecast_data.index[-1])
            }

            # Save forecast to database
            success = manager.save_model_forecasts(model_key, forecast_dict)

            if success:
                self.logger.info(f"Successfully saved forecast for {symbol} using model {model_key}")
            else:
                self.logger.error(f"Failed to save forecast for {symbol}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving forecast to database: {str(e)}")
            return False

    def load_forecast_from_db(self, db_manager: Any, symbol: str,
                              model_key: str) -> Optional[pd.Series]:
        try:
            self.logger.info(f"Loading forecast for {symbol} from model {model_key}")

            # Use the provided db_manager or the one saved in the class
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return None

            # Check if the model exists in the database
            model_info = manager.get_model_by_key(model_key)

            if model_info is None:
                self.logger.warning(f"Model with key {model_key} not found in database")
                return None

            # Get forecasts from database
            forecast_dict = manager.get_model_forecasts(model_key)

            if not forecast_dict:
                self.logger.warning(f"No forecasts found for model {model_key}")
                return None

            # Check if there are forecasts for the given symbol
            if symbol.upper() != forecast_dict.get('symbol', '').upper():
                self.logger.warning(f"Forecast for symbol {symbol} not found in model {model_key}")
                return None

            # Convert forecast dictionary to pd.Series
            try:
                forecast_data = forecast_dict.get('forecast_data', {})

                # Convert keys to datetime if they are dates
                index = []
                values = []

                for timestamp_str, value in forecast_data.items():
                    try:
                        # Try to convert to datetime
                        timestamp = pd.to_datetime(timestamp_str)
                    except:
                        # If conversion fails, use as is
                        timestamp = timestamp_str

                    index.append(timestamp)
                    values.append(float(value))

                # Create pandas Series with correct index
                forecast_series = pd.Series(values, index=index)

                # Sort by index
                forecast_series = forecast_series.sort_index()

                self.logger.info(f"Successfully loaded forecast with {len(forecast_series)} points for {symbol}")

                return forecast_series

            except Exception as e:
                self.logger.error(f"Error converting forecast data to Series: {str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading forecast from database: {str(e)}")
            return None

    def get_available_crypto_symbols(self) -> List[str]:

        self.logger.info("Getting available cryptocurrency symbols (fixed implementation)")

        # Return fixed list of available cryptocurrencies
        symbols = ["BTC", "ETH", "SOL"]

        self.logger.info(f"Found {len(symbols)} available cryptocurrency symbols")

        return symbols

    def get_last_update_time(self, db_manager: Any, symbol: str,
                             interval: str = '1d') -> Optional[datetime]:

        self.logger.info(f"Getting last update time for {symbol} with interval {interval}")

        if db_manager is None:
            self.logger.error("Database manager is not provided")
            return None

        try:
            # Check if the db_manager was passed during class initialization
            # if not, use the provided one
            db = self.db_manager if self.db_manager is not None else db_manager

            # Assume db_manager has a method to get the latest candlestick data
            latest_kline = db.get_latest_kline(symbol=symbol, interval=interval)

            if latest_kline is not None and hasattr(latest_kline, 'timestamp'):
                # If we got data and there is a timestamp
                self.logger.info(f"Last update time for {symbol} ({interval}): {latest_kline.timestamp}")
                return latest_kline.timestamp

            # If there is no direct method, try to get through processed candles
            klines_data = db.get_klines_processed(
                symbol=symbol,
                interval=interval,
                limit=1,  # Take only one (latest) candle
                sort_order="DESC"  # Sort by descending date
            )

            if klines_data is not None and not klines_data.empty:
                # Get the index of the last candle (which should be datetime)
                if isinstance(klines_data.index[0], datetime):
                    last_update = klines_data.index[0]
                else:
                    # If the index is not datetime, try to find a column with the timestamp
                    for col in ['timestamp', 'time', 'date', 'datetime']:
                        if col in klines_data.columns:
                            last_update = klines_data[col].iloc[0]
                            if not isinstance(last_update, datetime):
                                # Convert to datetime if it's not datetime
                                if isinstance(last_update, (int, float)):
                                    # Assume it's a UNIX timestamp in milliseconds or seconds
                                    if last_update > 1e11:  # If in milliseconds
                                        last_update = datetime.fromtimestamp(last_update / 1000)
                                    else:  # If in seconds
                                        last_update = datetime.fromtimestamp(last_update)
                                else:
                                    # If it's a string, try to parse
                                    try:
                                        from dateutil import parser
                                        last_update = parser.parse(str(last_update))
                                    except:
                                        self.logger.warning(f"Cannot parse datetime from {last_update}")
                                        continue
                            break
                    else:
                        self.logger.warning(f"No datetime column found in klines data for {symbol}")
                        return None

                self.logger.info(f"Last update time for {symbol} ({interval}): {last_update}")
                return last_update

            self.logger.warning(f"No data found for {symbol} with interval {interval}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting last update time for {symbol}: {str(e)}")
            return None

    def batch_process_symbols(self, db_manager: Any, symbols: List[str],
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              interval: str = '1d') -> Dict[str, Dict]:

        self.logger.info(f"Starting batch processing for {len(symbols)} symbols")

        if db_manager is None:
            self.logger.error("Database manager is not provided")
            return {"status": "error", "message": "Database manager is not provided"}

        # Initialize dictionary for results
        results = {}

        # Set default values for dates
        if end_date is None:
            end_date = datetime.now()
            self.logger.info(f"End date not provided, using current time: {end_date}")

        # Process each symbol
        for symbol in symbols:
            self.logger.info(f"Processing symbol: {symbol}")

            try:
                # Check if there is data for this symbol
                if not self._check_symbol_data_available(db_manager, symbol, interval):
                    self.logger.warning(f"No data available for {symbol}, skipping")
                    results[symbol] = {
                        "status": "error",
                        "message": f"No data available for {symbol}",
                        "timestamp": datetime.now()
                    }
                    continue

                # If the start date is not specified, get the last update date
                # and subtract a certain period (e.g., 365 days for daily data)
                if start_date is None:
                    last_update = self.get_last_update_time(db_manager, symbol, interval)
                    if last_update is not None:
                        if interval == '1d':
                            start_date = last_update - timedelta(days=365)  # Year of data for daily interval
                        elif interval == '1h':
                            start_date = last_update - timedelta(days=30)  # 30 days for hourly interval
                        elif interval in ['15m', '5m', '1m']:
                            start_date = last_update - timedelta(days=7)  # Week for minute intervals
                        else:
                            start_date = last_update - timedelta(days=180)  # Six months by default

                        self.logger.info(f"Calculated start date for {symbol}: {start_date}")
                    else:
                        self.logger.warning(f"Cannot determine last update time for {symbol}, using default")
                        # By default, take data for the last year
                        start_date = end_date - timedelta(days=365)

                # Load data for analysis
                data = self.load_crypto_data(
                    db_manager=db_manager,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=interval
                )

                if data is None or data.empty:
                    self.logger.warning(f"No data loaded for {symbol}, skipping")
                    results[symbol] = {
                        "status": "error",
                        "message": f"No data loaded for {symbol}",
                        "timestamp": datetime.now()
                    }
                    continue

                # Select target column for analysis (usually 'close')
                target_column = 'close'
                if target_column not in data.columns:
                    # Look for alternatives if 'close' is not present
                    possible_columns = ['Close', 'price', 'Price', 'value', 'Value']
                    for col in possible_columns:
                        if col in data.columns:
                            target_column = col
                            break
                    else:
                        # If there is no corresponding column, use the first column with numeric data
                        for col in data.columns:
                            if pd.api.types.is_numeric_dtype(data[col]):
                                target_column = col
                                break
                        else:
                            self.logger.error(f"No suitable numeric column found for {symbol}")
                            results[symbol] = {
                                "status": "error",
                                "message": f"No suitable numeric column found for {symbol}",
                                "timestamp": datetime.now()
                            }
                            continue

                self.logger.info(f"Using column '{target_column}' for analysis of {symbol}")

                # Run automatic forecasting
                forecast_result = self.forecaster.run_auto_forecast(
                    data=data[target_column],
                    test_size=0.2,  # 20% of data for testing
                    forecast_steps=24,  # Forecast 24 periods ahead
                    symbol=symbol
                )

                # Save forecasting results
                if forecast_result.get("status") == "success" and "model_key" in forecast_result:
                    model_key = forecast_result["model_key"]

                    # Save comprehensive model information to the database
                    if self.db_manager is not None:
                        try:
                            self.db_manager.save_complete_model(model_key, forecast_result.get("model_info", {}))
                            self.logger.info(f"Model for {symbol} saved to database with key {model_key}")
                        except Exception as db_error:
                            self.logger.error(f"Error saving model for {symbol} to database: {str(db_error)}")

                    # Add result to the general dictionary
                    results[symbol] = {
                        "status": "success",
                        "message": f"Successfully processed {symbol}",
                        "model_key": model_key,
                        "timestamp": datetime.now(),
                        **forecast_result
                    }
                else:
                    # If forecasting failed, add error information
                    results[symbol] = {
                        "status": "error",
                        "message": forecast_result.get("message", f"Error processing {symbol}"),
                        "timestamp": datetime.now()
                    }

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                results[symbol] = {
                    "status": "error",
                    "message": f"Exception: {str(e)}",
                    "timestamp": datetime.now()
                }

        # Add general statistics
        success_count = sum(1 for symbol, result in results.items() if result.get("status") == "success")
        error_count = len(symbols) - success_count

        summary = {
            "total_symbols": len(symbols),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(symbols) if symbols else 0,
            "processed_at": datetime.now()
        }

        results["_summary"] = summary

        self.logger.info(f"Batch processing completed. Success: {success_count}, Errors: {error_count}")

        return results

    def _check_symbol_data_available(self, db_manager: Any, symbol: str, interval: str) -> bool:

        try:
            # Check if this symbol is in the list of available symbols
            available_symbols = self.get_available_crypto_symbols()
            if symbol not in available_symbols:
                self.logger.warning(f"Symbol {symbol} not in available symbols list")
                return False

            # Check if there is at least some data for this symbol
            last_update = self.get_last_update_time(db_manager, symbol, interval)
            if last_update is None:
                self.logger.warning(f"No last update time for {symbol}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking data availability for {symbol}: {str(e)}")
            return False

    def preprocess_data(self, data: pd.Series, operations: List[Dict]) -> pd.Series:
        """Preprocessing of time series"""
        return self.transformer.apply_preprocessing_pipeline(data, operations)

    def check_stationarity(self, data: pd.Series) -> Dict:
        """Check stationarity of time series"""
        return self.forecaster._check_stationarity(data)

    def analyze_series(self, data: pd.Series) -> Dict:
        """Complete analysis of time series"""
        analysis = {}

        # Stationarity
        analysis['stationarity'] = self.check_stationarity(data)

        # Seasonality
        analysis['seasonality'] = self.analyzer.detect_seasonality(data)

        # Volatility
        analysis['volatility'] = self.transformer.extract_volatility(data).describe().to_dict()

        return analysis

    def find_optimal_model(self, data: pd.Series, seasonal: bool = False) -> Dict:
        """Find optimal model parameters"""
        return self.analyzer.find_optimal_params(data, seasonal=seasonal)

    def train_model(self, data: pd.Series, model_type: str = 'arima',
                    order: Tuple = None, seasonal_order: Tuple = None,
                    symbol: str = 'default') -> Dict:
        """Train ARIMA/SARIMA model"""
        if model_type == 'arima':
            if order is None:
                order = (1, 1, 1)  # Default values
            return self.modeler.fit_arima(data, order=order, symbol=symbol)
        elif model_type == 'sarima':
            if order is None or seasonal_order is None:
                # Auto-determine parameters
                params = self.find_optimal_model(data, seasonal=True)
                if params['status'] != 'success':
                    raise ValueError("Failed to determine optimal parameters")
                order = params['parameters']['order']
                seasonal_order = params['parameters']['seasonal_order']
            return self.modeler.fit_sarima(data, order=order,
                                           seasonal_order=seasonal_order,
                                           symbol=symbol)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def evaluate_model(self, model_key: str, test_data: pd.Series) -> Dict:
        """Evaluate model quality"""
        return self.evaluator.evaluate_model(model_key, test_data)

    def forecast(self, model_key: str, steps: int = 24,
                 with_intervals: bool = False, alpha: float = 0.05) -> Union[pd.Series, Dict]:
        """Generate forecast"""
        if with_intervals:
            return self.forecaster.forecast_with_intervals(model_key, steps=steps, alpha=alpha)
        else:
            return self.forecaster.forecast(model_key, steps=steps)

    def residual_analysis(self, model_key: str) -> Dict:
        """Analyze model residuals"""
        return self.evaluator.residual_analysis(model_key)

    def compare_models(self, model_keys: List[str], test_data: pd.Series) -> Dict:
        """Compare multiple models"""
        return self.evaluator.compare_models(model_keys, test_data)

    def save_model(self, model_key: str, path: str) -> bool:
        """Save model to disk"""
        return self.modeler.save_model(model_key, path)

    def load_model(self, model_key: str, path: str) -> bool:
        """Load model from disk"""
        return self.modeler.load_model(model_key, path)

    def full_pipeline(self, symbol: str, interval: str = '1d',
                      forecast_steps: int = 24) -> Dict:
        """Complete pipeline from data to forecast"""
        try:
            # 1. Load data
            data = self.load_crypto_data(db_manager=self.db_manager, symbol=symbol, timeframe=interval)
            if data.empty:
                return {"status": "error", "message": "No data loaded"}

            # Select target variable (closing)
            target = data['close']

            # 2. Data analysis
            analysis = self.analyze_series(target)

            # 3. Preprocessing
            operations = []
            if not analysis['stationarity']['is_stationary']:
                operations.append({"op": "diff", "order": 1})

            processed_data = self.preprocess_data(target, operations)

            # 4. Model selection and training
            # ARIMA
            arima_result = self.train_model(processed_data, 'arima', symbol=symbol)
            if arima_result['status'] != 'success':
                return arima_result

            # SARIMA (if seasonality is detected)
            sarima_result = None
            if analysis['seasonality']['has_seasonality']:
                sarima_result = self.train_model(
                    processed_data, 'sarima',
                    symbol=symbol,
                    seasonal_order=(1, 1, 1, analysis['seasonality']['primary_period'])
                )

            # 5. Model evaluation
            # Split into training and test sets
            train_size = int(len(processed_data) * 0.8)
            train_data = processed_data[:train_size]
            test_data = processed_data[train_size:]

            # Evaluate ARIMA
            arima_eval = self.evaluate_model(arima_result['model_key'], test_data)

            # Evaluate SARIMA (if available)
            sarima_eval = None
            if sarima_result and sarima_result['status'] == 'success':
                sarima_eval = self.evaluate_model(sarima_result['model_key'], test_data)

            # 6. Select the best model
            best_model_key = arima_result['model_key']
            if sarima_eval and sarima_eval['metrics']['rmse'] < arima_eval['metrics']['rmse']:
                best_model_key = sarima_result['model_key']

            # 7. Forecasting
            forecast = self.forecast(best_model_key, forecast_steps)

            # 8. Residual analysis
            residuals = self.residual_analysis(best_model_key)

            # 9. Save results
            result = {
                "status": "success",
                "symbol": symbol,
                "interval": interval,
                "data_analysis": analysis,
                "models": {
                    "arima": {
                        "model_key": arima_result['model_key'],
                        "evaluation": arima_eval
                    },
                    "sarima": {
                        "model_key": sarima_result['model_key'] if sarima_result else None,
                        "evaluation": sarima_eval
                    } if sarima_result else None
                },
                "best_model": best_model_key,
                "forecast": forecast,
                "residual_analysis": residuals,
                "timestamp": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in full pipeline: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def batch_process(self, symbols: List[str], interval: str = '1d') -> Dict:
        """Batch processing of multiple symbols"""
        results = {}

        for symbol in symbols:
            self.logger.info(f"Processing {symbol}...")
            try:
                result = self.full_pipeline(symbol, interval)
                results[symbol] = result

                # Save forecast if available
                if result['status'] == 'success' and 'forecast' in result:
                    self.save_forecast_to_db(
                        self.db_manager,
                        symbol,
                        result['forecast'],
                        result['best_model']
                    )

            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "message": str(e)
                }

        # Processing statistics
        success = sum(1 for r in results.values() if r['status'] == 'success')
        failed = len(symbols) - success

        return {
            "results": results,
            "summary": {
                "total": len(symbols),
                "success": success,
                "failed": failed,
                "success_rate": success / len(symbols) if symbols else 0
            }
        }

    def ensemble_forecast(self, data: pd.Series, models: List[str],
                          forecast_steps: int = 24, weights: Optional[List[float]] = None) -> Dict:
        """
        Create ensemble forecast from multiple models

        Args:
            data: Time series data
            models: List of model types to include ('arima', 'sarima', 'prophet', etc.)
            forecast_steps: Number of steps to forecast
            weights: List of weights for each model (optional)

        Returns:
            Dict: Ensemble forecast results
        """
        try:
            self.logger.info(f"Creating ensemble forecast with models: {models}")

            # Check if models list is valid
            valid_models = ['arima', 'sarima']
            if not all(m in valid_models for m in models):
                invalid = [m for m in models if m not in valid_models]
                self.logger.error(f"Invalid model types: {invalid}")
                return {"status": "error", "message": f"Invalid model types: {invalid}"}

            if weights is not None and len(weights) != len(models):
                self.logger.error(f"Number of weights ({len(weights)}) does not match number of models ({len(models)})")
                return {"status": "error", "message": "Weights and models count mismatch"}

            # If weights are not provided, use equal weights
            if weights is None:
                weights = [1 / len(models)] * len(models)

            # Normalize weights
            weights = [w / sum(weights) for w in weights]

            # Train each model and get forecasts
            forecasts = []
            model_keys = []

            for i, model_type in enumerate(models):
                self.logger.info(f"Training {model_type} model...")

                # Analyze data for seasonality
                seasonality = self.analyzer.detect_seasonality(data)
                has_seasonality = seasonality.get('has_seasonality', False)

                # Train model
                if model_type == 'arima':
                    result = self.train_model(data, 'arima')
                elif model_type == 'sarima' and has_seasonality:
                    result = self.train_model(
                        data, 'sarima',
                        seasonal_order=(1, 1, 1, seasonality.get('primary_period', 12))
                    )
                else:
                    # Skip this model
                    self.logger.warning(f"Skipping {model_type} model as it's not applicable")
                    continue

                if result['status'] == 'success':
                    # Get forecast
                    model_key = result['model_key']
                    model_keys.append(model_key)
                    forecast = self.forecast(model_key, steps=forecast_steps)
                    forecasts.append((forecast, weights[i]))
                else:
                    self.logger.warning(f"Failed to train {model_type} model: {result.get('message', 'Unknown error')}")

            if not forecasts:
                self.logger.error("No successful forecasts generated")
                return {"status": "error", "message": "No successful forecasts"}

            # Combine forecasts
            ensemble_forecast = None
            for forecast, weight in forecasts:
                if ensemble_forecast is None:
                    ensemble_forecast = forecast * weight
                else:
                    # Reindex to ensure indexes match
                    forecast = forecast.reindex(ensemble_forecast.index)
                    ensemble_forecast += forecast * weight

            return {
                "status": "success",
                "ensemble_forecast": ensemble_forecast,
                "component_models": model_keys,
                "weights": weights[:len(model_keys)]
            }

        except Exception as e:
            self.logger.error(f"Error creating ensemble forecast: {str(e)}")
            return {"status": "error", "message": str(e)}

    def visualize_forecast(self, historical_data: pd.Series, forecast_data: pd.Series,
                           save_path: Optional[str] = None) -> Dict:
        """
        Create visualization of forecast vs historical data

        Args:
            historical_data: Historical time series data
            forecast_data: Forecast time series data
            save_path: Path to save the visualization (optional)

        Returns:
            Dict: Status of visualization creation
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.figure import Figure

            self.logger.info("Creating forecast visualization")

            # Create figure
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)

            # Plot historical data
            ax.plot(historical_data.index, historical_data.values,
                    label='Historical', color='blue')

            # Plot forecast data
            ax.plot(forecast_data.index, forecast_data.values,
                    label='Forecast', color='red', linestyle='--')

            # Add forecast area
            ax.fill_between(forecast_data.index,
                            forecast_data.values * 0.95,
                            forecast_data.values * 1.05,
                            color='red', alpha=0.2)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()

            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Historical Data and Forecast')
            ax.legend()

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            if save_path:
                fig.savefig(save_path)
                self.logger.info(f"Visualization saved to {save_path}")

            return {
                "status": "success",
                "message": "Visualization created successfully",
                "figure": fig
            }

        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return {"status": "error", "message": str(e)}


def main():

        """Enhanced main function with better error handling but without saving anything to disk"""
        from datetime import datetime, timedelta

        # Configuration
        symbol = "BTC"
        timeframe = "1d"
        forecast_steps = 30  # Forecast for the next 30 days

        # Initialize TimeSeriesModels
        print("🔄 Initializing TimeSeriesModels...")
        model = TimeSeriesModels()
        db = model.db_manager

        if db is None:
            print("❌ Database manager is not configured.")
            return

        # Get list of available symbols
        print("🔄 Getting available symbols...")
        try:
            available_symbols = model.get_available_crypto_symbols()
            if not available_symbols:
                print("❌ No symbols available in the database.")
                return

            if symbol not in available_symbols:
                print(
                    f"⚠️ Symbol {symbol} not found in database. Available symbols: {', '.join(available_symbols[:5])}...")
                # Use the first available symbol instead
                symbol = available_symbols[0]
                print(f"🔄 Using {symbol} instead.")
        except Exception as e:
            print(f"❌ Error retrieving symbols: {e}")
            return

        # Get current date and calculate start date (2 years of historical data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        print(f"🔄 Loading {symbol} price data from {start_date.date()} to {end_date.date()}...")

        try:
            # Load data using the correct method from TimeSeriesModels
            df = model.load_crypto_data(
                db_manager=db,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
        except Exception as e:
            print(f"❌ Error while retrieving data: {e}")
            return

        if df is None or df.empty:
            print("❌ No data available for model training.")
            return

        # Print available columns to help debug
        print(f"📊 Available columns in the dataset: {', '.join(df.columns)}")

        # Check for price column with modified logic to handle preprocessed data
        price_columns = [
            "close", "Close", "price", "Price", "value", "Value",
            "original_close",  # Add the column from your database
            "close_log",  # Alternative preprocessed column
            "close_diff"  # Another alternative
        ]

        for col in price_columns:
            if col in df.columns:
                print(f"✅ Using '{col}' as price column")
                price_series = df[col]
                break
        else:
            # If no suitable column is found
            print("❌ No suitable price column found in data.")
            print(f"Available columns: {', '.join(df.columns)}")
            return

        # Data analysis and preprocessing
        print("🔄 Analyzing price data...")
        analysis = model.analyze_series(price_series)
        print(f"📊 Data analysis results:")
        print(f"  - Is stationary: {analysis['stationarity']['is_stationary']}")
        print(f"  - Has seasonality: {analysis['seasonality']['has_seasonality']}")
        if analysis['seasonality']['has_seasonality']:
            print(f"  - Primary seasonality period: {analysis['seasonality']['primary_period']}")
        print(f"  - Volatility (std): {analysis['volatility'].get('std', 'N/A')}")

        # Create train/test split for model validation
        train_size = int(len(price_series) * 0.8)
        train_data = price_series[:train_size]
        test_data = price_series[train_size:]
        print(f"📊 Data split: {train_size} training points, {len(test_data)} testing points")

        # Try ensemble forecast with multiple models
        print("🔄 Creating ensemble forecast...")
        ensemble_result = model.ensemble_forecast(
            data=train_data,
            models=['arima', 'sarima'],
            forecast_steps=forecast_steps
        )

        if ensemble_result.get('status') == 'success':
            ensemble_forecast = ensemble_result.get('ensemble_forecast')
            component_models = ensemble_result.get('component_models')
            print(f"✅ Ensemble forecast created using {len(component_models)} models")
            print(f"📈 First 5 forecast values: {ensemble_forecast.head().to_dict()}")

            # Print some forecast statistics instead of saving to CSV
            print(f"📊 Forecast statistics:")
            print(f"  - Forecast length: {len(ensemble_forecast)} points")
            print(f"  - Forecast period: {ensemble_forecast.index[0]} to {ensemble_forecast.index[-1]}")
            print(f"  - Forecast min: {ensemble_forecast.min()}")
            print(f"  - Forecast max: {ensemble_forecast.max()}")
            print(f"  - Forecast mean: {ensemble_forecast.mean()}")
        else:
            print(f"❌ Ensemble forecast failed: {ensemble_result.get('message', 'Unknown error')}")

            # Fallback to simple ARIMA forecast
            print("🔄 Falling back to standard ARIMA forecast...")
            forecast_result = model.forecaster.run_auto_forecast(
                data=price_series,
                test_size=0.2,
                forecast_steps=forecast_steps,
                symbol=symbol
            )

            if forecast_result.get("status") == "success" and "model_key" in forecast_result:
                model_key = forecast_result["model_key"]
                print(f"✅ Model created with key: {model_key}")

                # Get forecast
                forecast = model.forecast(model_key, steps=forecast_steps)
                if forecast is not None:
                    # Print forecast information instead of saving
                    print(f"📈 Forecast generated successfully")
                    print(f"  - Forecast length: {len(forecast)} points")
                    print(f"  - First 5 values: {forecast.head().to_dict()}")
                else:
                    print("❌ Failed to generate forecast.")
            else:
                print(f"❌ Auto forecast failed: {forecast_result.get('message', 'Unknown error')}")

        print("\n✅ Analysis completed.")

if __name__ == "__main__":
    main()