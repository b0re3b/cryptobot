from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd
from decimal import Decimal
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller, acf
from data.db import DatabaseManager
from utils.logger import CryptoLogger


class Forecaster:
    def __init__(self):
        self.models = {}
        self.logger = CryptoLogger('Forecaster')
        self.db_manager = DatabaseManager()

        # Import these classes only when needed to avoid circular imports
        from timeseriesmodels.TimeSeriesTransformer import TimeSeriesTransformer
        from timeseriesmodels.TimeSeriesAnalyzer import TimeSeriesAnalyzer
        from timeseriesmodels.ARIMAModeler import ARIMAModeler

        self.transformer = TimeSeriesTransformer()
        self.analyzer = TimeSeriesAnalyzer()
        self.modeler = ARIMAModeler()

    def _convert_decimal_series(self, series: pd.Series) -> pd.Series:
        """Convert decimal.Decimal objects to float for numpy compatibility"""
        try:
            # Check if series contains Decimal objects
            if series.dtype == 'object' and len(series) > 0:
                # Check first non-null value
                first_val = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
                if isinstance(first_val, Decimal):
                    self.logger.info("Converting Decimal objects to float for numpy compatibility")
                    # Convert Decimal to float
                    converted_series = series.apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                    # Ensure numeric dtype
                    converted_series = pd.to_numeric(converted_series, errors='coerce')
                    return converted_series

            # If not Decimal objects, ensure numeric dtype anyway
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            self.logger.warning(f"Error during decimal conversion: {str(e)}")
            # Fallback: try to convert to numeric
            return pd.to_numeric(series, errors='coerce')

    def check_stationarity(self, series: pd.Series) -> Dict:
        """Check if a time series is stationary using Augmented Dickey-Fuller test"""
        try:
            # Convert decimals to float if needed
            series = self._convert_decimal_series(series)

            # ADF test
            result = adfuller(series.dropna())
            adf_stat, p_value = result[0], result[1]

            # Interpret test results
            is_stationary = p_value < 0.05

            return {
                "is_stationary": is_stationary,
                "adf_statistic": adf_stat,
                "p_value": p_value,
                "critical_values": result[4]
            }
        except Exception as e:
            self.logger.error(f"Error in stationarity check: {str(e)}")
            # Default to non-stationary if test fails
            return {"is_stationary": False, "error": str(e)}

    def _create_forecast_index(self, data: pd.Series, steps: int, forecast_steps=24) -> pd.Index:
        """Create appropriate index for forecast values"""
        if isinstance(data.index, pd.DatetimeIndex):
            last_date = data.index[-1]

            # Try to determine frequency
            if len(data) >= 2:
                freq = pd.infer_freq(data.index)
                if freq:
                    forecast_index = pd.date_range(start=last_date + pd.Timedelta(seconds=1),
                                                   periods=steps,
                                                   freq=freq)
                else:
                    # If frequency detection fails, estimate median interval
                    time_diff = data.index[1:] - data.index[:-1]
                    median_diff = pd.Timedelta(np.median([d.total_seconds() for d in time_diff]), unit='s')
                    forecast_index = pd.date_range(start=last_date + median_diff,
                                                   periods=steps,
                                                   freq=median_diff)
            else:
                # Default to daily frequency if not enough data points
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                               periods=steps)
        else:
            # For numeric indices
            last_idx = data.index[-1]
            idx_diff = data.index[1] - data.index[0] if len(data) >= 2 else 1
            forecast_index = pd.RangeIndex(start=last_idx + idx_diff,
                                           stop=last_idx + idx_diff * (forecast_steps + 1),
                                           step=idx_diff)

        return forecast_index

    def forecast(self, model_key: str, steps: int = 24) -> pd.Series:
        """Generate forecast for a specified number of steps using a trained model"""
        self.logger.info(f"Starting forecast for model {model_key} with {steps} steps")

        # Check if model exists in memory
        if model_key not in self.models:
            # Try to load model from database
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Model {model_key} not found in memory, trying to load from database")
                    loaded = self.db_manager.load_complete_model(model_key)
                    if loaded:
                        self.logger.info(f"Model {model_key} successfully loaded from database")
                    else:
                        error_msg = f"Failed to load model {model_key} from database"
                        self.logger.error(error_msg)
                        return pd.Series([], dtype=float)
                except Exception as e:
                    error_msg = f"Error loading model {model_key} from database: {str(e)}"
                    self.logger.error(error_msg)
                    return pd.Series([], dtype=float)
            else:
                error_msg = f"Model {model_key} not found and no database manager provided"
                self.logger.error(error_msg)
                return pd.Series([], dtype=float)

        # Get the trained model
        try:
            model_info = self.models[model_key]
            fit_result = model_info.get("fit_result")
            metadata = model_info.get("metadata", {})

            if fit_result is None:
                error_msg = f"Model {model_key} has no fit result"
                self.logger.error(error_msg)
                return pd.Series([], dtype=float)

            # Determine model type for appropriate forecasting method
            model_type = metadata.get("model_type", "ARIMA")

            # Get last date from training data to build forecast index
            data_range = metadata.get("data_range", {})
            end_date_str = data_range.get("end")

            if end_date_str:
                try:
                    # Parse end date of training data
                    if isinstance(end_date_str, str):
                        try:
                            end_date = pd.to_datetime(end_date_str)
                        except:
                            end_date = datetime.now()  # Fallback to current date if parsing fails
                    else:
                        end_date = end_date_str
                except Exception as e:
                    self.logger.warning(f"Could not parse end date: {str(e)}, using current date")
                    end_date = datetime.now()
            else:
                self.logger.warning("No end date in metadata, using current date")
                end_date = datetime.now()

            # Generate forecast
            self.logger.info(f"Forecasting {steps} steps ahead with {model_type} model")

            # Use appropriate forecasting method based on model type
            if model_type == "ARIMA":
                # For ARIMA use direct forecast method
                forecast_result = fit_result.forecast(steps=steps)
            elif model_type == "SARIMA":
                # For SARIMA use get_forecast
                forecast_result = fit_result.get_forecast(steps=steps)
                forecast_result = forecast_result.predicted_mean
            else:
                error_msg = f"Unknown model type: {model_type}"
                self.logger.error(error_msg)
                return pd.Series([], dtype=float)

            # Get training data from fit_result to create forecast index
            if hasattr(fit_result.model, 'data') and hasattr(fit_result.model.data, 'orig_endog'):
                train_data = pd.Series(fit_result.model.data.orig_endog)
                if hasattr(fit_result.model.data, 'dates') and fit_result.model.data.dates is not None:
                    train_data.index = fit_result.model.data.dates
            else:
                # Create dummy series with numeric index as fallback
                train_data = pd.Series(range(100))  # Arbitrary length

            # Create index for forecast using the utility method
            forecast_index = self._create_forecast_index(train_data, steps)

            # Create Series with forecast and index
            forecast_series = pd.Series(forecast_result, index=forecast_index)

            # Save forecast to database if connection available
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Saving forecast for model {model_key} to database")
                    # Create forecast data dictionary for storage
                    forecast_data = {
                        "model_key": model_key,
                        "timestamp": datetime.now(),
                        "forecast_horizon": steps,
                        "values": forecast_series.to_dict(),
                        "start_date": forecast_index[0].isoformat() if isinstance(forecast_index[0], datetime) else str(
                            forecast_index[0]),
                        "end_date": forecast_index[-1].isoformat() if isinstance(forecast_index[-1], datetime) else str(
                            forecast_index[-1])
                    }
                    self.db_manager.save_model_forecasts(model_key, forecast_data)
                    self.logger.info(f"Forecast for model {model_key} saved successfully")
                except Exception as e:
                    self.logger.error(f"Error saving forecast to database: {str(e)}")

            self.logger.info(f"Forecast for model {model_key} completed successfully")
            return forecast_series

        except Exception as e:
            error_msg = f"Error during forecasting with model {model_key}: {str(e)}"
            self.logger.error(error_msg)
            return pd.Series([], dtype=float)

    def forecast_with_intervals(self, model_key: str, steps: int = 24, alpha: float = 0.05) -> Dict:
        """Generate forecast with confidence intervals"""
        self.logger.info(f"Starting forecast with intervals for model {model_key}, steps={steps}, alpha={alpha}")

        # Validate alpha
        if alpha <= 0 or alpha >= 1:
            error_msg = f"Invalid alpha value ({alpha}). Must be between 0 and 1."
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Check model availability
        if model_key not in self.models:
            # Try to load model from database
            if self.db_manager is not None:
                try:
                    model_loaded = self.db_manager.load_complete_model(model_key)
                    if not model_loaded:
                        error_msg = f"Model {model_key} not found in database"
                        self.logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                    self.logger.info(f"Model {model_key} loaded from database")
                except Exception as e:
                    error_msg = f"Error loading model {model_key} from database: {str(e)}"
                    self.logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
            else:
                error_msg = f"Model {model_key} not found and no database manager available"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

        try:
            # Get model information
            model_info = self.models.get(model_key)
            if not model_info:
                error_msg = f"Model {model_key} information not available"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

            fit_result = model_info.get("fit_result")
            if not fit_result:
                error_msg = f"Fit result for model {model_key} not available"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

            metadata = model_info.get("metadata", {})
            model_type = metadata.get("model_type", "unknown")

            # Get data transformation information
            transformations = None
            if self.db_manager is not None:
                try:
                    transformations = self.db_manager.get_data_transformations(model_key)
                    if transformations:
                        self.logger.info(f"Found data transformations for model {model_key}")
                except Exception as e:
                    self.logger.warning(f"Error getting data transformations: {str(e)}")

            # Generate forecast with confidence intervals
            try:
                # Generate forecast with confidence intervals
                forecast_result = fit_result.get_forecast(steps=steps)

                # Get prediction values and intervals
                predicted_mean = forecast_result.predicted_mean
                confidence_intervals = forecast_result.conf_int(alpha=alpha)

                # Get training data from model for index creation
                if hasattr(fit_result.model, 'data') and hasattr(fit_result.model.data, 'orig_endog'):
                    train_data = pd.Series(fit_result.model.data.orig_endog)
                    if hasattr(fit_result.model.data, 'dates') and fit_result.model.data.dates is not None:
                        train_data.index = fit_result.model.data.dates
                else:
                    # Create dummy series with numeric index
                    train_data = pd.Series(range(100))  # Arbitrary length

                # Create time indices for forecast using the utility method
                forecast_index = self._create_forecast_index(train_data, steps)

                # Create Series for forecast and intervals
                forecast_series = pd.Series(predicted_mean, index=forecast_index)
                lower_bound = pd.Series(confidence_intervals.iloc[:, 0].values, index=forecast_index)
                upper_bound = pd.Series(confidence_intervals.iloc[:, 1].values, index=forecast_index)

                # Apply inverse transformation if needed
                if transformations:
                    try:
                        transform_method = transformations.get("method")
                        transform_param = transformations.get("lambda_param")

                        if transform_method:
                            self.logger.info(f"Applying inverse transformation: {transform_method}")
                            forecast_series = self.transformer.inverse_transform(
                                forecast_series,
                                method=transform_method,
                                lambda_param=transform_param
                            )
                            lower_bound = self.transformer.inverse_transform(
                                lower_bound,
                                method=transform_method,
                                lambda_param=transform_param
                            )
                            upper_bound = self.transformer.inverse_transform(
                                upper_bound,
                                method=transform_method,
                                lambda_param=transform_param
                            )
                    except Exception as e:
                        self.logger.warning(f"Error during inverse transformation: {str(e)}")

                # Format forecast results
                forecast_data = {
                    "forecast": forecast_series.tolist(),
                    "lower_bound": lower_bound.tolist(),
                    "upper_bound": upper_bound.tolist(),
                    "indices": [str(idx) for idx in forecast_index],
                    "confidence_level": 1.0 - alpha
                }

                # Save forecast results to database
                if self.db_manager is not None:
                    try:
                        forecast_db_data = {
                            "model_key": model_key,
                            "forecast_timestamp": datetime.now(),
                            "steps": steps,
                            "alpha": alpha,
                            "forecast_data": {
                                "values": forecast_series.tolist(),
                                "lower_bound": lower_bound.tolist(),
                                "upper_bound": upper_bound.tolist(),
                                "indices": [str(idx) for idx in forecast_index],
                                "confidence_level": 1.0 - alpha
                            }
                        }
                        self.db_manager.save_model_forecasts(model_key, forecast_db_data)
                        self.logger.info(f"Forecast results for model {model_key} saved to database")
                    except Exception as e:
                        self.logger.warning(f"Error saving forecast results to database: {str(e)}")

                # Complete result
                result = {
                    "status": "success",
                    "model_key": model_key,
                    "model_type": model_type,
                    "forecast_timestamp": datetime.now().isoformat(),
                    "steps": steps,
                    "alpha": alpha,
                    "forecast_data": forecast_data
                }

                self.logger.info(f"Forecast with intervals for model {model_key} completed successfully")
                return result

            except Exception as e:
                error_msg = f"Error during forecasting: {str(e)}"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during forecast with intervals: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def run_auto_forecast(self, data: pd.Series, test_size: float = 0.2,
                          forecast_steps: int = 24, symbol: str = 'auto') -> Dict:
        """Automatically analyze time series, fit optimal model, and generate forecasts"""
        self.logger.info(f"Starting auto forecasting process for symbol: {symbol}")

        # Convert decimal objects to float if needed
        data = self._convert_decimal_series(data)

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before auto forecasting.")
            data = data.dropna()

        if len(data) < 30:  # Minimum number of points for meaningful analysis
            error_msg = "Not enough data points for auto forecasting (min 30 required)"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "model_key": None,
                "forecasts": None,
                "performance": None
            }

        try:
            # Generate unique key for model
            model_key = f"{symbol}_auto_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # 1. Split data into training and test sets
            train_size = int(len(data) * (1 - test_size))
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            self.logger.info(f"Split data: train={len(train_data)}, test={len(test_data)}")

            # 2. Check stationarity and apply transformations
            stationarity_check = self.check_stationarity(train_data)
            transformations = []
            transformed_data = train_data.copy()

            # If series is non-stationary, apply transformations
            if not stationarity_check["is_stationary"]:
                self.logger.info("Time series is non-stationary. Applying transformations.")

                # a) Logarithmic transformation if all data > 0
                if all(train_data > 0):
                    self.logger.info("Applying log transformation")
                    transformed_data = np.log(transformed_data)
                    transformations.append({"op": "log"})

                    # Check stationarity after log transformation
                    log_stationary = self.check_stationarity(transformed_data)["is_stationary"]

                    if not log_stationary:
                        # b) If still non-stationary, apply differencing
                        self.logger.info("Series still non-stationary. Applying differencing.")
                        transformed_data = self.transformer.difference_series(transformed_data, order=1)
                        transformations.append({"op": "diff", "order": 1})
                else:
                    # If there are negative values, apply differencing directly
                    self.logger.info("Series contains non-positive values. Applying differencing directly.")
                    transformed_data = self.transformer.difference_series(train_data, order=1)
                    transformations.append({"op": "diff", "order": 1})

            # 3. Detect seasonality
            # Heuristic for detecting seasonality through autocorrelation
            seasonal = False
            seasonal_period = None

            if len(transformed_data) > 50:  # Enough data for seasonality analysis
                max_lag = min(len(transformed_data) // 2, 365)  # Limit maximum lag
                acf_vals = acf(transformed_data, nlags=max_lag, fft=True)

                # Look for peaks in autocorrelation (potential seasonal periods)
                potential_periods = []

                # Check typical periods for financial data
                for period in [7, 14, 30, 90, 180, 365]:
                    if period < len(acf_vals):
                        if acf_vals[period] > 0.3:  # Significant autocorrelation
                            potential_periods.append((period, acf_vals[period]))

                if potential_periods:
                    # Choose period with strongest autocorrelation
                    potential_periods.sort(key=lambda x: x[1], reverse=True)
                    seasonal = True
                    seasonal_period = potential_periods[0][0]
                    self.logger.info(f"Detected seasonality with period: {seasonal_period}")

            # 4. Find optimal model parameters
            if seasonal and seasonal_period:
                # For seasonal series
                optimal_params = self.analyzer.find_optimal_params(
                    transformed_data,
                    max_p=3, max_d=1, max_q=3,
                    seasonal=True,
                    seasonal_period=seasonal_period
                )
            else:
                # For non-seasonal series
                optimal_params = self.analyzer.find_optimal_params(
                    transformed_data,
                    max_p=5, max_d=1, max_q=5,
                    seasonal=False
                )

            if optimal_params.get("status") == "error":
                self.logger.error(f"Parameter search failed: {optimal_params['message']}")
                return {
                    "status": "error",
                    "message": f"Parameter search failed: {optimal_params['message']}",
                    "model_key": None,
                    "forecasts": None,
                    "performance": None
                }

            # 5. Train model with optimal parameters
            model_info = None

            if seasonal and seasonal_period:
                # SARIMA model
                order = optimal_params["parameters"]["order"]
                seasonal_order = optimal_params["parameters"]["seasonal_order"]

                fit_result = self.modeler.fit_sarima(
                    transformed_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    symbol=symbol
                )

                model_type = "SARIMA"
            else:
                # ARIMA model
                order = optimal_params["parameters"]["order"]

                fit_result = self.modeler.fit_arima(
                    transformed_data,
                    order=order,
                    symbol=symbol
                )

                model_type = "ARIMA"

            if fit_result.get("status") == "error":
                self.logger.error(f"Model fitting failed: {fit_result['message']}")
                return {
                    "status": "error",
                    "message": f"Model fitting failed: {fit_result['message']}",
                    "model_key": model_key,
                    "forecasts": None,
                    "performance": None
                }

            # Save trained model key
            model_key = fit_result["model_key"]
            model_info = fit_result["model_info"]

            # 6. Generate forecast
            model_obj = self.models[model_key]["fit_result"]

            # Forecast test data (if available)
            test_performance = None
            if len(test_data) > 0:
                try:
                    # Forecast test period for evaluation
                    test_forecast = model_obj.forecast(len(test_data))

                    # For seasonal models or models with differencing,
                    # need to account for initial values in inverse transformation
                    if "diff" in [t.get("op") for t in transformations]:
                        # Simplified approach - compare trends
                        test_performance = {
                            "mse": mean_squared_error(test_data.values, test_forecast),
                            "rmse": np.sqrt(mean_squared_error(test_data.values, test_forecast)),
                            "mae": mean_absolute_error(test_data.values, test_forecast)
                        }

                        # Calculate MAPE if no zero values
                        if all(test_data != 0):
                            mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                            test_performance["mape"] = mape
                    else:
                        # For models without differencing, compare directly
                        test_performance = {
                            "mse": mean_squared_error(test_data.values, test_forecast),
                            "rmse": np.sqrt(mean_squared_error(test_data.values, test_forecast)),
                            "mae": mean_absolute_error(test_data.values, test_forecast)
                        }

                        # Calculate MAPE if no zero values
                        if all(test_data != 0):
                            mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                            test_performance["mape"] = mape
                except Exception as e:
                    self.logger.error(f"Error during test forecast: {str(e)}")
                    test_performance = {"error": str(e)}

            # 7. Forecast future periods
            try:
                future_forecast = model_obj.forecast(forecast_steps)

                # Create forecast index
                forecast_index = self._create_forecast_index(data, forecast_steps)

                # Create Series for forecast with proper index
                future_forecast = pd.Series(future_forecast, index=forecast_index)

                # 8. Apply inverse transformations (if transformations were applied)
                for transform in reversed(transformations):
                    if transform["op"] == "diff":
                        # For inverse differencing, need initial value
                        # Use last value of original series
                        last_orig_value = data.iloc[-1]
                        future_forecast = future_forecast.cumsum() + last_orig_value
                    elif transform["op"] == "log":
                        future_forecast = np.exp(future_forecast)

                self.logger.info(f"Forecast completed: {len(future_forecast)} steps")
            except Exception as e:
                self.logger.error(f"Error during future forecast: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error during future forecast: {str(e)}",
                    "model_key": model_key,
                    "forecasts": None,
                    "performance": test_performance
                }

            # 9. Save results to database (if available)
            if self.db_manager is not None:
                try:
                    # Collect all data for storage
                    forecast_data = {
                        "future_forecast": future_forecast.to_dict(),
                        "forecast_steps": forecast_steps,
                        "forecast_date": datetime.now().isoformat()
                    }

                    # Save forecast
                    self.db_manager.save_model_forecasts(model_key, forecast_data)

                    if test_performance:
                        # Save performance metrics
                        self.db_manager.save_model_metrics(model_key, test_performance)

                    # Save data transformation information
                    self.db_manager.save_data_transformations(model_key, {"transformations": transformations})

                    self.logger.info(f"Model {model_key} and forecast data saved to database")
                except Exception as db_error:
                    self.logger.error(f"Error saving to database: {str(db_error)}")

            # 10. Format result
            result = {
                "status": "success",
                "message": f"{model_type} model trained and forecast completed successfully",
                "model_key": model_key,
                "model_info": model_info,
                "transformations": transformations,
                "forecasts": {
                    "values": future_forecast.to_dict(),
                    "steps": forecast_steps
                },
                "performance": test_performance
            }

            self.logger.info(f"Auto forecast completed successfully for symbol: {symbol}")
            return result

        except Exception as e:
            self.logger.error(f"Error during auto forecasting: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during auto forecasting: {str(e)}",
                "model_key": model_key if 'model_key' in locals() else None,
                "forecasts": None,
                "performance": None
            }