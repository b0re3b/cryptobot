from datetime import datetime
from typing import Dict, Union, List, Optional, Tuple
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
        """
            Конвертує об'єкти decimal.Decimal у серії pandas у float для сумісності з numpy.

            Аргументи:
                series (pd.Series): Вхідна серія, яка може містити об'єкти типу Decimal.

            Повертає:
                pd.Series: Серія, де значення типу Decimal конвертовані у float,
                           інші значення залишені без змін.
                           Якщо конвертація неможлива, повертає серію з числовими значеннями
                           (помилки при конвертації замінюються на NaN).
            """
        try:
            if series.dtype == 'object' and len(series) > 0:
                first_val = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
                if isinstance(first_val, Decimal):
                    self.logger.info("Converting Decimal objects to float for numpy compatibility")
                    converted_series = series.apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                    converted_series = pd.to_numeric(converted_series, errors='coerce')
                    return converted_series

            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            self.logger.warning(f"Error during decimal conversion: {str(e)}")
            return pd.to_numeric(series, errors='coerce')

    def _validate_input_data(self, data: pd.Series, min_length: int = 30) -> Dict:
        """
            Перевіряє та попередньо обробляє вхідні дані для моделювання часових рядів.

            Виконує:
            - Перевірку на порожні або None дані,
            - Конвертацію у pandas Series, якщо потрібно,
            - Конвертацію об'єктів Decimal у float,
            - Перевірку мінімальної довжини даних,
            - Видалення NaN значень (з повторною перевіркою довжини),
            - Перевірку індексу та конвертацію у DatetimeIndex, якщо можливо,
            - Сортування даних за індексом, якщо індекс datetime і не впорядкований.

            Аргументи:
                data (pd.Series): Вхідні дані для перевірки (можуть бути у вигляді списку або іншої ітерації).
                min_length (int, необов'язковий): Мінімальна кількість точок даних, необхідна для роботи (за замовчуванням 30).

            Повертає:
                dict: Результат у форматі словника:
                    - 'status' (str): "success" або "error",
                    - 'data' (pd.Series, якщо успішно): Оброблені дані,
                    - 'message' (str, якщо помилка): Опис помилки.
            """
        try:
            if data is None or len(data) == 0:
                return {"status": "error", "message": "Input data is empty or None"}

            # Convert to pandas Series if needed
            if not isinstance(data, pd.Series):
                try:
                    data = pd.Series(data)
                except Exception as e:
                    return {"status": "error", "message": f"Cannot convert input to pandas Series: {str(e)}"}

            # Convert decimal objects
            data = self._convert_decimal_series(data)

            # Check for minimum length
            if len(data) < min_length:
                return {"status": "error",
                        "message": f"Not enough data points (min {min_length} required, got {len(data)})"}

            # Handle NaN values
            if data.isnull().any():
                self.logger.warning(f"Data contains {data.isnull().sum()} NaN values. Cleaning data.")
                data = data.dropna()

                if len(data) < min_length:
                    return {"status": "error", "message": f"After removing NaN values, not enough data points remain"}

            # Ensure index is datetime if possible
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except:
                    # If conversion fails, create a simple range index
                    data.index = pd.RangeIndex(len(data))

            # Sort by index
            if isinstance(data.index, pd.DatetimeIndex) and not data.index.is_monotonic_increasing:
                data = data.sort_index()

            return {"status": "success", "data": data}

        except Exception as e:
            return {"status": "error", "message": f"Error validating input data: {str(e)}"}

    def _create_forecast_index(self, data: pd.Series, steps: int) -> pd.Index:
        """
           Створює відповідний індекс для прогнозованих значень.

           Якщо вхідні дані мають datetime-індекс, намагається вивести частоту інтервалів.
           Якщо частоту визначити не вдалося — використовує медіану інтервалів.
           Якщо індекс числовий — створює RangeIndex із збереженням кроку індексації.

           Аргументи:
               data (pd.Series): Серія вхідних часових даних із індексом.
               steps (int): Кількість кроків прогнозу.

           Повертає:
               pd.Index: Індекс для прогнозованих значень із відповідною частотою або кроком.
           """
        if isinstance(data.index, pd.DatetimeIndex):
            last_date = data.index[-1]

            # Try to determine frequency
            if len(data) >= 2:
                freq = pd.infer_freq(data.index)
                if freq:
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(seconds=1),
                        periods=steps,
                        freq=freq
                    )
                else:
                    # Estimate median interval
                    time_diff = data.index[1:] - data.index[:-1]
                    median_diff = pd.Timedelta(np.median([d.total_seconds() for d in time_diff]), unit='s')
                    forecast_index = pd.date_range(
                        start=last_date + median_diff,
                        periods=steps,
                        freq=median_diff
                    )
            else:
                # Default to daily frequency
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=steps
                )
        else:
            # For numeric indices
            last_idx = data.index[-1]
            idx_diff = data.index[1] - data.index[0] if len(data) >= 2 else 1
            forecast_index = pd.RangeIndex(
                start=last_idx + idx_diff,
                stop=last_idx + idx_diff * (steps + 1),
                step=idx_diff
            )

        return forecast_index

    def _ensure_model_loaded(self, model_key: str) -> Dict:
        """
            Переконується, що модель із заданим ключем завантажена в пам'ять.

            Якщо моделі немає в пам'яті, намагається завантажити її з бази даних.
            Перевіряє наявність основних компонентів моделі (наприклад, fit_result).
            Повертає статус успіху або помилки з повідомленням.

            Аргументи:
                model_key (str): Унікальний ключ моделі.

            Повертає:
                dict: Результат з полями:
                    - 'status' (str): 'success' або 'error'
                    - 'message' (str): Опис помилки, якщо є
            """
        if model_key not in self.models:
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Loading model {model_key} from database")

                    # Load model data from database
                    model_data = self.db_manager.load_model_data(model_key)
                    if not model_data:
                        return {"status": "error", "message": f"No model data found for {model_key}"}

                    # Store the loaded model in memory
                    self.models[model_key] = model_data

                    # Verify the model is properly loaded
                    if model_key not in self.models:
                        return {"status": "error", "message": f"Model {model_key} failed to load into memory"}

                    # Verify required components exist
                    model_info = self.models[model_key]
                    if "fit_result" not in model_info or model_info["fit_result"] is None:
                        return {"status": "error", "message": f"Model {model_key} has no fit_result"}

                    self.logger.info(f"Model {model_key} successfully loaded and verified")
                    return {"status": "success"}

                except Exception as e:
                    self.logger.error(f"Error loading model {model_key}: {str(e)}")
                    return {"status": "error", "message": f"Error loading model {model_key}: {str(e)}"}
            else:
                return {"status": "error", "message": f"Model {model_key} not found and no database manager available"}

        # Model already in memory - verify it's valid
        try:
            model_info = self.models[model_key]
            if "fit_result" not in model_info or model_info["fit_result"] is None:
                return {"status": "error", "message": f"Model {model_key} in memory but has no fit_result"}
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": f"Error validating model {model_key} in memory: {str(e)}"}

    def register_model(self, model_key: str, fit_result, metadata: Dict = None) -> None:
        """
           Реєструє навчену модель у пам'яті.

           Зберігає результат навчання, метадані, час створення та ключ моделі.
           Виводить інформаційне повідомлення про успішну реєстрацію або повідомлення про помилку.

           Аргументи:
               model_key (str): Унікальний ідентифікатор моделі.
               fit_result: Результат навчання моделі (наприклад, об'єкт SARIMAXResults).
               metadata (dict, optional): Додаткова інформація про модель (тип, параметри, тощо).
           """
        try:
            self.models[model_key] = {
                "fit_result": fit_result,
                "metadata": metadata or {},
                "created_at": datetime.now(),
                "model_key": model_key
            }
            self.logger.info(f"Model {model_key} registered in memory")
        except Exception as e:
            self.logger.error(f"Error registering model {model_key}: {str(e)}")

    def get_model_info(self, model_key: str) -> Dict:
        """
            Отримує інформацію про модель за її ключем із обробкою помилок.

            Завантажує модель із бази даних, якщо вона ще не в пам'яті.
            Повертає основні характеристики моделі (AIC, BIC, параметри, порядок тощо).
            У разі помилки — повертає мінімальну доступну інформацію.

            Аргументи:
                model_key (str): Унікальний ідентифікатор моделі.

            Повертає:
                dict: Результат у форматі:
                    - status (str): 'success' або 'error'
                    - model_info (dict, якщо success): основна інформація про модель
                    - message (str, якщо error): опис помилки
            """
        try:
            # Ensure model is loaded
            load_result = self._ensure_model_loaded(model_key)
            if load_result["status"] == "error":
                return {"status": "error", "message": load_result["message"]}

            model_info = self.models[model_key]
            fit_result = model_info.get("fit_result")
            metadata = model_info.get("metadata", {})

            if fit_result is None:
                return {"status": "error", "message": f"Model {model_key} has no fit result"}

            # Extract model information safely
            try:
                model_summary = {
                    "model_key": model_key,
                    "model_type": metadata.get("model_type", "ARIMA"),
                    "aic": getattr(fit_result, 'aic', None),
                    "bic": getattr(fit_result, 'bic', None),
                    "hqic": getattr(fit_result, 'hqic', None),
                    "llf": getattr(fit_result, 'llf', None),
                    "created_at": model_info.get("created_at"),
                    "parameters": {}
                }

                # Get model parameters safely
                if hasattr(fit_result, 'params'):
                    model_summary["parameters"] = fit_result.params.to_dict()
                elif hasattr(fit_result, 'model') and hasattr(fit_result.model, 'params'):
                    model_summary["parameters"] = fit_result.model.params.to_dict()

                # Get order information
                if hasattr(fit_result, 'model'):
                    if hasattr(fit_result.model, 'order'):
                        model_summary["order"] = fit_result.model.order
                    if hasattr(fit_result.model, 'seasonal_order'):
                        model_summary["seasonal_order"] = fit_result.model.seasonal_order

                return {"status": "success", "model_info": model_summary}

            except Exception as e:
                self.logger.warning(f"Error extracting detailed model info for {model_key}: {str(e)}")
                return {
                    "status": "success",
                    "model_info": {
                        "model_key": model_key,
                        "model_type": metadata.get("model_type", "ARIMA"),
                        "created_at": model_info.get("created_at"),
                        "note": "Limited info due to extraction error"
                    }
                }

        except Exception as e:
            return {"status": "error", "message": f"Error getting model info: {str(e)}"}

    def _generate_model_forecast(self, model_key: str, steps: int, alpha: Optional[float] = None) -> Dict:
        """
            Основний метод генерації прогнозу з покращеною обробкою помилок.

            Повертає точковий прогноз або прогноз з довірчими інтервалами (за потреби).
            Перевіряє наявність моделі у пам’яті та її готовність до прогнозування.

            Аргументи:
                model_key (str): Унікальний ідентифікатор моделі.
                steps (int): Кількість кроків для прогнозування.
                alpha (float, optional): Рівень значущості для довірчого інтервалу (наприклад, 0.05 для 95%).

            Повертає:
                dict: Результат у форматі:
                    - status (str): 'success' або 'error'
                    - forecast (Series): Прогнозовані значення
                    - lower_bound (Series, optional): Нижній межа довірчого інтервалу
                    - upper_bound (Series, optional): Верхня межа довірчого інтервалу
                    - confidence_level (float, optional): Рівень довіри
                    - message (str, якщо статус 'error'): Опис помилки
            """
        try:
            if model_key not in self.models:
                return {"status": "error", "message": f"Model {model_key} not found in memory"}

            model_info = self.models[model_key]
            fit_result = model_info.get("fit_result")
            metadata = model_info.get("metadata", {})

            if fit_result is None:
                return {"status": "error", "message": f"Model {model_key} has no fit result"}

            model_type = metadata.get("model_type", "ARIMA")

            # Generate forecast based on model type and requirements
            if alpha is not None:
                # Generate forecast with confidence intervals
                try:
                    forecast_result = fit_result.get_forecast(steps=steps)
                    predicted_mean = forecast_result.predicted_mean
                    confidence_intervals = forecast_result.conf_int(alpha=alpha)

                    return {
                        "status": "success",
                        "forecast": predicted_mean,
                        "lower_bound": confidence_intervals.iloc[:, 0],
                        "upper_bound": confidence_intervals.iloc[:, 1],
                        "confidence_level": 1.0 - alpha
                    }
                except Exception as e:
                    return {"status": "error", "message": f"Error generating forecast with intervals: {str(e)}"}
            else:
                # Generate point forecast only
                try:
                    if hasattr(fit_result, 'get_forecast'):
                        forecast_result = fit_result.get_forecast(steps=steps)
                        forecast_values = forecast_result.predicted_mean
                    elif hasattr(fit_result, 'forecast'):
                        forecast_values = fit_result.forecast(steps=steps)
                    else:
                        return {"status": "error", "message": "Model does not support forecasting"}

                    return {
                        "status": "success",
                        "forecast": forecast_values
                    }
                except Exception as e:
                    return {"status": "error", "message": f"Error generating point forecast: {str(e)}"}

        except Exception as e:
            return {"status": "error", "message": f"Error in forecast generation: {str(e)}"}

    def _apply_transformations(self, data: pd.Series, reverse: bool = False,
                               transformations: Optional[Dict] = None) -> pd.Series:
        """
            Застосовує або інвертує перетворення даних.

            Підтримує різні методи трансформації, наприклад диференціювання.
            Якщо вказано `reverse=True`, застосовує зворотну трансформацію.

            Аргументи:
                data (pd.Series): Вхідні часові ряди.
                reverse (bool): Чи застосовувати обернене перетворення.
                transformations (dict, optional): Параметри перетворення, наприклад:
                    {
                        "method": "diff" або "diff_2",
                        "lambda_param": float (для Box-Cox, якщо реалізовано)
                    }

            Повертає:
                pd.Series: Перетворені або зворотно перетворені дані.
            """
        if not transformations or not transformations.get("method"):
            return data

        try:
            method = transformations["method"]

            if reverse:
                # Apply inverse transformation
                return self.transformer.inverse_transform(
                    data,
                    method=method,
                    lambda_param=transformations.get("lambda_param")
                )
            else:
                # Apply forward transformation
                if method == "diff":
                    return self.transformer.difference_series(data, order=1)
                elif method == "diff_2":
                    temp = self.transformer.difference_series(data, order=1)
                    return self.transformer.difference_series(temp, order=1)
                # Add other transformation methods as needed

        except Exception as e:
            self.logger.warning(f"Error during transformation: {str(e)}")

        return data

    def _save_forecast_to_db(self, model_key: str, forecast_data: Dict) -> bool:
        """Save forecast results to database"""
        if self.db_manager is None:
            return False

        try:
            self.db_manager.save_model_forecasts(model_key, forecast_data)
            self.logger.info(f"Forecast for model {model_key} saved to database")
            return True
        except Exception as e:
            self.logger.warning(f"Error saving forecast to database: {str(e)}")
            return False

    def forecast(self, model_key: str, steps: int = 24) -> pd.Series:
        """
            Генерує точковий прогноз для вказаної моделі на задану кількість кроків вперед.

            Метод виконує:
            - Перевірку завантаження моделі
            - Генерацію прогнозу
            - Створення індексу майбутніх значень
            - Застосування зворотних трансформацій (якщо задано)
            - Збереження прогнозу в базу даних

            Аргументи:
                model_key (str): Унікальний ідентифікатор моделі.
                steps (int): Кількість майбутніх кроків прогнозу (по замовчуванню — 24).

            Повертає:
                pd.Series: Ряд прогнозованих значень з часовим індексом або порожній Series у разі помилки.
            """
        self.logger.info(f"Starting forecast for model {model_key} with {steps} steps")

        # Ensure model is loaded
        load_result = self._ensure_model_loaded(model_key)
        if load_result["status"] == "error":
            self.logger.error(load_result["message"])
            return pd.Series([], dtype=float, name='forecast_error')

        # Generate forecast
        forecast_result = self._generate_model_forecast(model_key, steps)
        if forecast_result["status"] == "error":
            self.logger.error(forecast_result["message"])
            return pd.Series([], dtype=float, name='forecast_error')

        try:
            # Get model metadata for index creation
            model_info = self.models[model_key]
            metadata = model_info.get("metadata", {})

            # Create training data series for index generation
            fit_result = model_info["fit_result"]

            # Try different ways to get training data
            train_data = None
            if hasattr(fit_result, 'model') and hasattr(fit_result.model, 'data'):
                if hasattr(fit_result.model.data, 'orig_endog'):
                    train_data = pd.Series(fit_result.model.data.orig_endog)
                    if hasattr(fit_result.model.data, 'dates') and fit_result.model.data.dates is not None:
                        try:
                            train_data.index = fit_result.model.data.dates
                        except:
                            pass
                elif hasattr(fit_result.model.data, 'endog'):
                    train_data = pd.Series(fit_result.model.data.endog)

            # Fallback if we can't get training data
            if train_data is None or len(train_data) == 0:
                train_data = pd.Series(range(100))
                self.logger.warning("Using fallback training data for index generation")

            # Create forecast index and series
            forecast_index = self._create_forecast_index(train_data, steps)
            forecast_series = pd.Series(forecast_result["forecast"], index=forecast_index, name='forecast')

            # Apply inverse transformations if needed
            transformations = None
            if self.db_manager is not None:
                try:
                    transformations = self.db_manager.get_data_transformations(model_key)
                except Exception as e:
                    self.logger.debug(f"No transformations found for {model_key}: {str(e)}")

            if transformations:
                try:
                    forecast_series = self._apply_transformations(forecast_series, reverse=True,
                                                                  transformations=transformations)
                except Exception as e:
                    self.logger.warning(f"Error applying inverse transformations: {str(e)}")

            # Save to database
            try:
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
                self._save_forecast_to_db(model_key, forecast_data)
            except Exception as e:
                self.logger.warning(f"Error saving forecast to database: {str(e)}")

            self.logger.info(f"Forecast for model {model_key} completed successfully")
            return forecast_series

        except Exception as e:
            self.logger.error(f"Error processing forecast results: {str(e)}")
            return pd.Series([], dtype=float, name='forecast_error')

    def forecast_with_intervals(self, model_key: str, steps: int = 24, alpha: float = 0.05) -> Dict:
        """
            Генерує прогноз з довірчими інтервалами для вказаної моделі.

            Метод виконує:
            - Перевірку параметра alpha
            - Перевірку завантаження моделі
            - Генерацію прогнозу з довірчими інтервалами
            - Створення індексу майбутніх значень
            - Застосування зворотних трансформацій (якщо задано)
            - Збереження результатів у базу даних

            Аргументи:
                model_key (str): Унікальний ідентифікатор моделі.
                steps (int): Кількість кроків прогнозу.
                alpha (float): Рівень значущості для довірчого інтервалу (наприклад, 0.05 для 95%).

            Повертає:
                dict: Словник з результатами:
                    - status (str): 'success' або 'error'
                    - forecast_data (dict): Прогноз, межі довіри, індекси
                    - forecast_timestamp (str): Час створення прогнозу
                    - model_key (str), model_type (str), steps (int), alpha (float)
                    - message (str, якщо статус 'error'): Опис помилки
            """
        self.logger.info(f"Starting forecast with intervals for model {model_key}, steps={steps}, alpha={alpha}")

        # Validate alpha
        if alpha <= 0 or alpha >= 1:
            return {"status": "error", "message": f"Invalid alpha value ({alpha}). Must be between 0 and 1."}

        # Ensure model is loaded
        load_result = self._ensure_model_loaded(model_key)
        if load_result["status"] == "error":
            return {"status": "error", "message": load_result["message"]}

        # Generate forecast with intervals
        forecast_result = self._generate_model_forecast(model_key, steps, alpha=alpha)
        if forecast_result["status"] == "error":
            return {"status": "error", "message": forecast_result["message"]}

        try:
            # Get model info
            model_info = self.models[model_key]
            metadata = model_info.get("metadata", {})
            model_type = metadata.get("model_type", "unknown")

            # Create training data for index generation
            fit_result = model_info["fit_result"]
            train_data = None

            if hasattr(fit_result, 'model') and hasattr(fit_result.model, 'data'):
                if hasattr(fit_result.model.data, 'orig_endog'):
                    train_data = pd.Series(fit_result.model.data.orig_endog)
                    if hasattr(fit_result.model.data, 'dates') and fit_result.model.data.dates is not None:
                        try:
                            train_data.index = fit_result.model.data.dates
                        except:
                            pass

            if train_data is None:
                train_data = pd.Series(range(100))

            # Create forecast index
            forecast_index = self._create_forecast_index(train_data, steps)

            # Create series
            forecast_series = pd.Series(forecast_result["forecast"], index=forecast_index)
            lower_bound = pd.Series(forecast_result["lower_bound"].values, index=forecast_index)
            upper_bound = pd.Series(forecast_result["upper_bound"].values, index=forecast_index)

            # Apply inverse transformations if needed
            transformations = None
            if self.db_manager is not None:
                try:
                    transformations = self.db_manager.get_data_transformations(model_key)
                except:
                    pass

            if transformations:
                try:
                    forecast_series = self._apply_transformations(forecast_series, reverse=True,
                                                                  transformations=transformations)
                    lower_bound = self._apply_transformations(lower_bound, reverse=True,
                                                              transformations=transformations)
                    upper_bound = self._apply_transformations(upper_bound, reverse=True,
                                                              transformations=transformations)
                except Exception as e:
                    self.logger.warning(f"Error applying inverse transformations: {str(e)}")

            # Format results
            forecast_data = {
                "forecast": forecast_series.tolist(),
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist(),
                "indices": [str(idx) for idx in forecast_index],
                "confidence_level": 1.0 - alpha
            }

            # Save to database
            try:
                forecast_db_data = {
                    "model_key": model_key,
                    "forecast_timestamp": datetime.now(),
                    "steps": steps,
                    "alpha": alpha,
                    "forecast_data": forecast_data
                }
                self._save_forecast_to_db(model_key, forecast_db_data)
            except Exception as e:
                self.logger.warning(f"Error saving forecast to database: {str(e)}")

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
            return {"status": "error", "message": f"Error processing forecast with intervals: {str(e)}"}

    def check_stationarity(self, series: pd.Series) -> Dict:
        """Check if a time series is stationary using Augmented Dickey-Fuller test"""
        try:
            series = self._convert_decimal_series(series)
            result = adfuller(series.dropna())
            adf_stat, p_value = result[0], result[1]
            is_stationary = p_value < 0.05

            return {
                "is_stationary": is_stationary,
                "adf_statistic": adf_stat,
                "p_value": p_value,
                "critical_values": result[4]
            }
        except Exception as e:
            self.logger.error(f"Error in stationarity check: {str(e)}")
            return {"is_stationary": False, "error": str(e)}

    def auto_determine_order(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[
        int, int, int]:
        """
            Автоматично визначає оптимальні параметри порядку (p, d, q) для моделі ARIMA.

            Параметри:
            ----------
            data : pd.Series
                Одновимірний часовий ряд, для якого визначається порядок моделі.
            max_p : int, optional
                Максимальне значення параметра p (кількість авторегресивних лагів).
            max_d : int, optional
                Максимальне значення параметра d (кількість диференціювань).
            max_q : int, optional
                Максимальне значення параметра q (кількість лагів скользячого середнього).

            Повертає:
            --------
            Tuple[int, int, int]
                Кортеж з трьох цілих чисел — (p, d, q). У випадку помилки повертається (1, 1, 1).
            """
        try:
            optimal_params = self.analyzer.find_optimal_params(
                data, max_p=max_p, max_d=max_d, max_q=max_q, seasonal=False
            )

            if isinstance(optimal_params, dict) and "parameters" in optimal_params:
                return optimal_params["parameters"].get("order", (1, 1, 1))
            elif isinstance(optimal_params, dict) and "order" in optimal_params:
                return optimal_params["order"]
            else:
                return (1, 1, 1)  # Default fallback

        except Exception as e:
            self.logger.warning(f"Error in auto order determination: {str(e)}")
            return (1, 1, 1)

    def train_model(self, data: pd.Series, model_type: str = 'arima',
                    order: Optional[Tuple[int, int, int]] = None,
                    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                    symbol: str = 'auto') -> Dict:
        """
            Уніфікований метод для тренування моделей ARIMA або SARIMA.

            Параметри:
            ----------
            data : pd.Series
                Одновимірний часовий ряд, який використовується для навчання моделі.
            model_type : str, optional
                Тип моделі для тренування ('arima' або 'sarima').
            order : Tuple[int, int, int], optional
                Параметри (p, d, q) для ARIMA/SARIMA. Якщо не задано — обирається автоматично.
            seasonal_order : Tuple[int, int, int, int], optional
                Сезонні параметри (P, D, Q, s) для SARIMA. Необов'язкові.
            symbol : str, optional
                Назва символьного ідентифікатора моделі (наприклад, тикер криптовалюти).

            Повертає:
            --------
            Dict
                Словник із результатом тренування, включаючи `status`, `model_key`, `model_info` і тип моделі.
            """
        try:
            # Validate input data
            validation_result = self._validate_input_data(data)
            if validation_result["status"] == "error":
                return validation_result

            data = validation_result["data"]

            # Auto-determine order if not provided
            if order is None:
                order = self.auto_determine_order(data)

            # Generate model key
            model_key = f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Train based on model type
            if model_type.lower() == 'sarima' and seasonal_order:
                fit_result = self.modeler.fit_sarima(data, order=order,
                                                     seasonal_order=seasonal_order,
                                                     symbol=symbol)
            else:
                fit_result = self.modeler.fit_arima(data, order=order, symbol=symbol)

            if isinstance(fit_result, dict) and fit_result.get("status") == "success":
                # Register the model in memory
                actual_model_key = fit_result.get("model_key", model_key)
                if "model_info" in fit_result and "fit_result" in fit_result["model_info"]:
                    self.register_model(
                        actual_model_key,
                        fit_result["model_info"]["fit_result"],
                        {"model_type": model_type.upper(), "symbol": symbol}
                    )

                return {
                    "status": "success",
                    "model_key": actual_model_key,
                    "model_info": fit_result.get("model_info"),
                    "model_type": model_type.upper()
                }
            else:
                return {"status": "error", "message": "Model training failed"}

        except Exception as e:
            return {"status": "error", "message": f"Error training model: {str(e)}"}

    def run_auto_forecast(self, data: pd.Series, test_size: float = 0.2,
                          forecast_steps: int = 24, symbol: str = 'auto') -> Dict:
        """
            Повністю автоматизований процес прогнозування:
            - Аналіз станціонарності
            - Трансформація ряду
            - Визначення сезонності
            - Вибір та тренування моделі (ARIMA/SARIMA)
            - Прогноз та оцінка якості

            Параметри:
            ----------
            data : pd.Series
                Часовий ряд для моделювання і прогнозу.
            test_size : float, optional
                Частка даних для тестування (за замовчуванням 0.2).
            forecast_steps : int, optional
                Кількість кроків майбутнього прогнозу.
            symbol : str, optional
                Назва або ідентифікатор ряду/моделі.

            Повертає:
            --------
            Dict
                Словник із результатами прогнозування: статус, тип моделі, ключ моделі, перетворення,
                майбутні прогнози та метрики точності.
            """
        self.logger.info(f"Starting auto forecasting process for symbol: {symbol}")

        # Validate input data
        validation_result = self._validate_input_data(data)
        if validation_result["status"] == "error":
            return {
                "status": "error",
                "message": validation_result["message"],
                "model_key": None,
                "forecasts": None,
                "performance": None
            }

        data = validation_result["data"]
        model_key = None

        try:
            # Generate unique model key
            model_key = f"{symbol}_auto_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Split data
            train_size = int(len(data) * (1 - test_size))
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            self.logger.info(f"Split data: train={len(train_data)}, test={len(test_data)}")

            # Check stationarity and apply transformations
            stationarity_check = self.check_stationarity(train_data)
            transformed_data = train_data.copy()
            transform_method = None

            if not stationarity_check["is_stationary"]:
                self.logger.info("Time series is non-stationary. Applying differencing.")
                try:
                    transformed_data = self.transformer.difference_series(train_data, order=1)
                    transform_method = "diff"

                    # Check if second-order differencing is needed
                    if not self.check_stationarity(transformed_data)["is_stationary"] and len(transformed_data) > 10:
                        transformed_data = self.transformer.difference_series(transformed_data, order=1)
                        transform_method = "diff_2"

                except Exception as e:
                    return {"status": "error", "message": f"Data transformation failed: {str(e)}"}

            # Detect seasonality
            seasonal = False
            seasonal_period = None

            if len(transformed_data) > 50:
                try:
                    max_lag = min(len(transformed_data) // 2, 365)
                    acf_vals = acf(transformed_data, nlags=max_lag, fft=True)

                    for period in [7, 14, 30, 90, 180, 365]:
                        if period < len(acf_vals) and acf_vals[period] > 0.3:
                            seasonal = True
                            seasonal_period = period
                            break

                except Exception as e:
                    self.logger.warning(f"Error in seasonality detection: {str(e)}")

            # Find optimal parameters
            if seasonal and seasonal_period:
                optimal_params = self.analyzer.find_optimal_params(
                    transformed_data, max_p=3, max_d=1, max_q=3, seasonal=True
                )
                model_type = "SARIMA"
            else:
                optimal_params = self.analyzer.find_optimal_params(
                    transformed_data, max_p=5, max_d=1, max_q=5, seasonal=False
                )
                model_type = "ARIMA"

            if isinstance(optimal_params, dict) and optimal_params.get("status") == "error":
                return {"status": "error", "message": f"Parameter optimization failed: {optimal_params.get('message')}"}

            # Extract order parameters
            if isinstance(optimal_params, dict):
                if "parameters" in optimal_params:
                    order = optimal_params["parameters"].get("order", (1, 1, 1))
                    seasonal_order = optimal_params["parameters"].get("seasonal_order")
                else:
                    order = optimal_params.get("order", (1, 1, 1))
                    seasonal_order = optimal_params.get("seasonal_order")
            else:
                order = (1, 1, 1)
                seasonal_order = None

            # Train model
            if seasonal and seasonal_period:
                if seasonal_order is None:
                    seasonal_order = (1, 1, 1, seasonal_period)
                fit_result = self.modeler.fit_sarima(transformed_data, order=order,
                                                     seasonal_order=seasonal_order, symbol=symbol)
            else:
                fit_result = self.modeler.fit_arima(transformed_data, order=order, symbol=symbol)

            if not (isinstance(fit_result, dict) and fit_result.get("status") == "success"):
                return {"status": "error", "message": "Model fitting failed"}

            model_key = fit_result.get("model_key", model_key)
            model_info = fit_result.get("model_info")

            # Save transformations to database
            if transform_method and self.db_manager is not None:
                try:
                    transformations = {"method": transform_method}
                    self.db_manager.save_data_transformations(model_key, transformations)
                except Exception as e:
                    self.logger.warning(f"Error saving transformations: {str(e)}")

            # Generate test forecast for performance evaluation
            test_performance = None
            if len(test_data) > 0:
                try:
                    test_forecast = self.forecast(model_key, steps=len(test_data))

                    if len(test_forecast) > 0:
                        # Align forecasts with test data
                        min_len = min(len(test_data), len(test_forecast))
                        test_actual = test_data.iloc[:min_len].values
                        test_pred = test_forecast.iloc[:min_len].values

                        test_performance = {
                            "mse": float(mean_squared_error(test_actual, test_pred)),
                            "rmse": float(np.sqrt(mean_squared_error(test_actual, test_pred))),
                            "mae": float(mean_absolute_error(test_actual, test_pred))
                        }

                        if all(test_actual != 0):
                            mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
                            test_performance["mape"] = float(mape)

                except Exception as e:
                    self.logger.error(f"Error during test forecast: {str(e)}")
                    test_performance = {"error": str(e)}

            # Generate future forecast
            future_forecast = self.forecast(model_key, steps=forecast_steps)

            if len(future_forecast) == 0:
                return {"status": "error", "message": "Future forecast generation failed"}

            # Save performance metrics
            if test_performance and "error" not in test_performance and self.db_manager is not None:
                try:
                    self.db_manager.save_model_metrics(model_key, test_performance)
                except Exception as e:
                    self.logger.warning(f"Error saving metrics: {str(e)}")

            # Format result
            result = {
                "status": "success",
                "message": f"{model_type} model trained and forecast completed successfully",
                "model_key": model_key,
                "model_info": model_info,
                "model_type": model_type,
                "transformations": {"method": transform_method} if transform_method else None,
                "forecasts": {
                    "values": future_forecast.to_dict(),
                    "steps": forecast_steps,
                    "start_date": future_forecast.index[0].isoformat() if isinstance(future_forecast.index[0],
                                                                                     datetime) else str(
                        future_forecast.index[0]),
                    "end_date": future_forecast.index[-1].isoformat() if isinstance(future_forecast.index[-1],
                                                                                    datetime) else str(
                        future_forecast.index[-1])
                },
                "performance": test_performance,
                "seasonality": {"detected": seasonal, "period": seasonal_period} if seasonal else None
            }

            self.logger.info(f"Auto forecast completed successfully for symbol: {symbol}")
            return result

        except Exception as e:
            self.logger.error(f"Error during auto forecasting: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during auto forecasting: {str(e)}",
                "model_key": model_key,
                "forecasts": None,
                "performance": None
            }

