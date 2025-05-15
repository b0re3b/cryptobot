import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
from data.db import DatabaseManager

class TimeSeriesModels:

    def __init__(self, log_level=logging.INFO):

        self.db_manager = DatabaseManager()
        self.models = {}  # Словник для збереження навчених моделей
        self.transformations = {}  # Словник для збереження параметрів трансформацій

        # Налаштування логування
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Якщо немає обробників логів, додаємо обробник для виведення в консоль
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("TimeSeriesModels initialized")

    def difference_series(self, data: pd.Series, order: int = 1) -> pd.Series:

        if order < 1:
            self.logger.warning("Differencing order must be at least 1, using order=1 instead")
            order = 1

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before differencing.")
            data = data.dropna()

        if len(data) <= order:
            self.logger.error(f"Not enough data points for {order}-order differencing")
            return pd.Series([], index=pd.DatetimeIndex([]))

        # Застосовуємо послідовне диференціювання
        diff_data = data.copy()
        for i in range(order):
            diff_data = diff_data.diff().dropna()

            # Перевіряємо чи залишились дані після диференціювання
            if len(diff_data) == 0:
                self.logger.error(f"No data left after {i + 1}-order differencing")
                return pd.Series([], index=pd.DatetimeIndex([]))

        self.logger.info(
            f"Applied {order}-order differencing. Original length: {len(data)}, Result length: {len(diff_data)}")

        return diff_data

    def find_optimal_params(self, data: pd.Series, max_p: int = 5, max_d: int = 2,
                            max_q: int = 5, seasonal: bool = False) -> Dict:

        self.logger.info("Starting optimal parameters search")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before parameter search.")
            data = data.dropna()

        if len(data) < 10:
            self.logger.error("Not enough data points for parameter search")
            return {
                "status": "error",
                "message": "Not enough data points for parameter search",
                "parameters": None,
                "model_info": None
            }

        try:
            # Використовуємо auto_arima для автоматичного пошуку параметрів
            if seasonal:
                # Для SARIMA: визначаємо можливий сезонний період
                # Типові значення: щотижнева (7), щомісячна (30/31), квартальна (4)
                seasonal_periods = [7, 12, 24, 30, 365]

                # Автовизначення сезонного періоду, якщо достатньо даних
                if len(data) >= 2 * max(seasonal_periods):
                    # Аналізуємо автокореляцію для виявлення можливого сезонного періоду
                    from statsmodels.tsa.stattools import acf
                    acf_values = acf(data, nlags=max(seasonal_periods))

                    # Шукаємо піки в ACF, які можуть вказувати на сезонність
                    potential_seasons = []
                    for period in seasonal_periods:
                        if period < len(acf_values) and acf_values[period] > 0.2:  # Поріг кореляції
                            potential_seasons.append((period, acf_values[period]))

                    # Вибираємо період з найсильнішою автокореляцією
                    if potential_seasons:
                        potential_seasons.sort(key=lambda x: x[1], reverse=True)
                        seasonal_period = potential_seasons[0][0]
                        self.logger.info(f"Detected potential seasonal period: {seasonal_period}")
                    else:
                        # За замовчуванням
                        seasonal_period = 7  # Тижнева сезонність для фінансових даних
                        self.logger.info(f"No strong seasonality detected, using default: {seasonal_period}")
                else:
                    seasonal_period = 7
                    self.logger.info(f"Not enough data for seasonal detection, using default: {seasonal_period}")

                # Запускаємо auto_arima з урахуванням сезонності
                model = auto_arima(
                    data,
                    start_p=0, max_p=max_p,
                    start_q=0, max_q=max_q,
                    max_d=max_d,
                    start_P=0, max_P=2,
                    start_Q=0, max_Q=2,
                    max_D=1,
                    m=seasonal_period,
                    seasonal=True,
                    trace=True,  # Виведення інформації про процес пошуку
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,  # Покроковий пошук для прискорення
                    information_criterion='aic',  # AIC або BIC як критерій
                    random_state=42
                )

                order = model.order
                seasonal_order = model.seasonal_order

                result = {
                    "status": "success",
                    "message": "Optimal parameters found",
                    "parameters": {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "seasonal_period": seasonal_period
                    },
                    "model_info": {
                        "aic": model.aic(),
                        "bic": model.bic(),
                        "model_type": "SARIMA"
                    }
                }
            else:
                # Для несезонної ARIMA
                model = auto_arima(
                    data,
                    start_p=0, max_p=max_p,
                    start_q=0, max_q=max_q,
                    max_d=max_d,
                    seasonal=False,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    information_criterion='aic',
                    random_state=42
                )

                order = model.order

                result = {
                    "status": "success",
                    "message": "Optimal parameters found",
                    "parameters": {
                        "order": order
                    },
                    "model_info": {
                        "aic": model.aic(),
                        "bic": model.bic(),
                        "model_type": "ARIMA"
                    }
                }

            self.logger.info(f"Found optimal parameters: {result['parameters']}")
            return result

        except Exception as e:
            self.logger.error(f"Error during parameter search: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during parameter search: {str(e)}",
                "parameters": None,
                "model_info": None
            }

    def fit_arima(self, data: pd.Series, order: Tuple[int, int, int],
                  symbol: str = 'default') -> Dict:

        self.logger.info(f"Starting ARIMA model training with order {order} for symbol {symbol}")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before fitting.")
            data = data.dropna()

        if len(data) < sum(order) + 1:
            error_msg = f"Not enough data points for ARIMA{order}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "model_key": None,
                "model_info": None
            }

        try:
            # Генеруємо унікальний ключ для моделі
            model_key = f"{symbol}_arima_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Тренування моделі ARIMA
            model = ARIMA(data, order=order)
            fit_result = model.fit()

            # Збираємо метадані моделі
            model_metadata = {
                "model_type": "ARIMA",
                "symbol": symbol,
                "timestamp": datetime.now(),
                "data_range": {
                    "start": data.index[0].isoformat() if isinstance(data.index[0], datetime) else str(data.index[0]),
                    "end": data.index[-1].isoformat() if isinstance(data.index[-1], datetime) else str(data.index[-1]),
                    "length": len(data)
                },
                "model_key": model_key
            }

            # Збираємо параметри моделі
            model_parameters = {
                "order": order,
                "training_info": {
                    "convergence": True if fit_result.mle_retvals.get('converged', False) else False,
                    "iterations": fit_result.mle_retvals.get('iterations', None)
                }
            }

            # Збираємо основну статистику моделі
            model_stats = {
                "aic": fit_result.aic,
                "bic": fit_result.bic,
                "aicc": fit_result.aicc if hasattr(fit_result, 'aicc') else None,
                "log_likelihood": fit_result.llf
            }

            # Зберігаємо модель в словнику
            self.models[model_key] = {
                "model": model,
                "fit_result": fit_result,
                "metadata": model_metadata,
                "parameters": model_parameters,
                "stats": model_stats
            }

            if self.db_manager is not None:
                try:
                    # Зберігаємо метадані
                    self.db_manager.save_model_metadata(model_key, model_metadata)

                    # Зберігаємо параметри
                    self.db_manager.save_model_parameters(model_key, model_parameters)

                    # Зберігаємо двійкове представлення моделі
                    import pickle
                    model_binary = pickle.dumps(fit_result)
                    self.db_manager.save_model_binary(model_key, model_binary)

                    self.logger.info(f"Model {model_key} saved to database")
                except Exception as db_error:
                    self.logger.error(f"Error saving model to database: {str(db_error)}")

            self.logger.info(f"ARIMA model {model_key} trained successfully")

            return {
                "status": "success",
                "message": "ARIMA model trained successfully",
                "model_key": model_key,
                "model_info": {
                    "metadata": model_metadata,
                    "parameters": model_parameters,
                    "stats": model_stats
                }
            }

        except Exception as e:
            self.logger.error(f"Error during ARIMA model training: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during ARIMA model training: {str(e)}",
                "model_key": None,
                "model_info": None
            }

    def fit_sarima(self, data: pd.Series, order: Tuple[int, int, int],
                   seasonal_order: Tuple[int, int, int, int], symbol: str = 'default') -> Dict:

        self.logger.info(
            f"Starting SARIMA model training with order {order}, seasonal_order {seasonal_order} for symbol {symbol}")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before fitting.")
            data = data.dropna()

        # Перевіряємо мінімальну необхідну довжину даних (p+d+q+P+D+Q+s > елементів)
        min_required = sum(order) + sum(seasonal_order[:-1]) + 2 * seasonal_order[-1]
        if len(data) < min_required:
            error_msg = f"Not enough data points for SARIMA{order}x{seasonal_order}. Need at least {min_required}, got {len(data)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "model_key": None,
                "model_info": None
            }

        try:
            # Генеруємо унікальний ключ для моделі
            model_key = f"{symbol}_sarima_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Тренування моделі SARIMA використовуючи SARIMAX
            model = SARIMAX(
                data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            # Встановлюємо опції підгонки для кращої збіжності
            fit_options = {
                'disp': False,  # Не виводити інформацію про ітерації
                'maxiter': 200,  # Максимальна кількість ітерацій
                'method': 'lbfgs'  # Метод оптимізації
            }

            fit_result = model.fit(**fit_options)

            # Збираємо метадані моделі
            model_metadata = {
                "model_type": "SARIMA",
                "symbol": symbol,
                "timestamp": datetime.now(),
                "data_range": {
                    "start": data.index[0].isoformat() if isinstance(data.index[0], datetime) else str(data.index[0]),
                    "end": data.index[-1].isoformat() if isinstance(data.index[-1], datetime) else str(data.index[-1]),
                    "length": len(data)
                },
                "model_key": model_key
            }

            # Збираємо параметри моделі
            model_parameters = {
                "order": order,
                "seasonal_order": seasonal_order,
                "training_info": {
                    "convergence": True if fit_result.mle_retvals.get('converged', False) else False,
                    "iterations": fit_result.mle_retvals.get('iterations', None)
                }
            }

            # Збираємо основну статистику моделі
            model_stats = {
                "aic": fit_result.aic,
                "bic": fit_result.bic,
                "aicc": fit_result.aicc if hasattr(fit_result, 'aicc') else None,
                "log_likelihood": fit_result.llf
            }

            # Зберігаємо модель в словнику
            self.models[model_key] = {
                "model": model,
                "fit_result": fit_result,
                "metadata": model_metadata,
                "parameters": model_parameters,
                "stats": model_stats
            }

            if self.db_manager is not None:
                try:
                    # Зберігаємо метадані
                    self.db_manager.save_model_metadata(model_key, model_metadata)

                    # Зберігаємо параметри
                    self.db_manager.save_model_parameters(model_key, model_parameters)

                    # Зберігаємо двійкове представлення моделі
                    import pickle
                    model_binary = pickle.dumps(fit_result)
                    self.db_manager.save_model_binary(model_key, model_binary)

                    self.logger.info(f"Model {model_key} saved to database")
                except Exception as db_error:
                    self.logger.error(f"Error saving model to database: {str(db_error)}")

            self.logger.info(f"SARIMA model {model_key} trained successfully")

            return {
                "status": "success",
                "message": "SARIMA model trained successfully",
                "model_key": model_key,
                "model_info": {
                    "metadata": model_metadata,
                    "parameters": model_parameters,
                    "stats": model_stats
                }
            }

        except Exception as e:
            self.logger.error(f"Error during SARIMA model training: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during SARIMA model training: {str(e)}",
                "model_key": None,
                "model_info": None
            }

    def forecast(self, model_key: str, steps: int = 24) -> pd.Series:

        self.logger.info(f"Starting forecast for model {model_key} with {steps} steps")

        # Перевіряємо наявність моделі в пам'яті
        if model_key not in self.models:
            # Якщо моделі немає в пам'яті, спробуємо завантажити її з БД
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Model {model_key} not found in memory, trying to load from database")
                    loaded = self.db_manager.load_сomplete_model(model_key)
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

        # Отримуємо навчену модель
        try:
            model_info = self.models[model_key]
            fit_result = model_info.get("fit_result")
            metadata = model_info.get("metadata", {})

            if fit_result is None:
                error_msg = f"Model {model_key} has no fit result"
                self.logger.error(error_msg)
                return pd.Series([], dtype=float)

            # Визначаємо тип моделі для вибору методу прогнозування
            model_type = metadata.get("model_type", "ARIMA")

            # Отримання останньої дати з даних навчання для побудови індексу прогнозу
            data_range = metadata.get("data_range", {})
            end_date_str = data_range.get("end")

            if end_date_str:
                try:
                    # Парсимо дату кінця навчальних даних
                    if isinstance(end_date_str, str):
                        try:
                            end_date = pd.to_datetime(end_date_str)
                        except:
                            end_date = datetime.now()  # Якщо парсинг не вдався
                    else:
                        end_date = end_date_str
                except Exception as e:
                    self.logger.warning(f"Could not parse end date: {str(e)}, using current date")
                    end_date = datetime.now()
            else:
                self.logger.warning("No end date in metadata, using current date")
                end_date = datetime.now()

            # Виконуємо прогнозування
            self.logger.info(f"Forecasting {steps} steps ahead with {model_type} model")

            # Для ARIMA і SARIMA моделей використовуємо різні методи прогнозування
            if model_type == "ARIMA":
                # Для ARIMA використовуємо прямий метод forecast
                forecast_result = fit_result.forecast(steps=steps)
            elif model_type == "SARIMA":
                # Для SARIMA використовуємо get_forecast
                forecast_result = fit_result.get_forecast(steps=steps)
                forecast_result = forecast_result.predicted_mean
            else:
                error_msg = f"Unknown model type: {model_type}"
                self.logger.error(error_msg)
                return pd.Series([], dtype=float)

            # Створюємо індекс для прогнозу
            # Припускаємо, що частота даних відповідає останній різниці в індексі
            try:
                # Спроба визначити частоту даних
                if "seasonal_order" in model_info.get("parameters", {}):
                    # Якщо це сезонна модель, використовуємо сезонний період
                    seasonal_period = model_info["parameters"]["seasonal_order"][3]
                    # Для денних даних
                    if seasonal_period == 7:
                        freq = 'D'  # день
                    elif seasonal_period in [12, 24]:
                        freq = 'H'  # година
                    elif seasonal_period == 30 or seasonal_period == 31:
                        freq = 'D'  # день
                    elif seasonal_period == 365:
                        freq = 'D'  # день
                    else:
                        freq = 'D'  # за замовчуванням
                else:
                    # За замовчуванням для несезонних моделей
                    freq = 'D'

                # Створюємо DatetimeIndex для прогнозу
                forecast_index = pd.date_range(start=end_date + timedelta(days=1),
                                               periods=steps,
                                               freq=freq)
            except Exception as e:
                self.logger.warning(f"Could not create date index: {str(e)}, using numeric index")
                forecast_index = range(steps)

            # Створюємо Series з прогнозом та індексом
            forecast_series = pd.Series(forecast_result, index=forecast_index)

            # Зберігаємо прогноз у БД, якщо є підключення
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Saving forecast for model {model_key} to database")
                    # Створюємо словник з прогнозом для збереження
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

    def evaluate_model(self, model_key: str, test_data: pd.Series) -> Dict:

        self.logger.info(f"Starting evaluation of model {model_key}")

        # Перевіряємо, чи є дані для тестування
        if test_data is None or len(test_data) == 0:
            error_msg = "Test data is empty or None"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "metrics": None
            }

        # Перевіряємо наявність моделі в пам'яті
        if model_key not in self.models:
            # Якщо моделі немає в пам'яті, спробуємо завантажити її з БД
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Model {model_key} not found in memory, trying to load from database")
                    loaded = self.db_manager.load_complete_model(model_key)
                    if loaded:
                        self.logger.info(f"Model {model_key} successfully loaded from database")
                    else:
                        error_msg = f"Failed to load model {model_key} from database"
                        self.logger.error(error_msg)
                        return {
                            "status": "error",
                            "message": error_msg,
                            "metrics": None
                        }
                except Exception as e:
                    error_msg = f"Error loading model {model_key} from database: {str(e)}"
                    self.logger.error(error_msg)
                    return {
                        "status": "error",
                        "message": error_msg,
                        "metrics": None
                    }
            else:
                error_msg = f"Model {model_key} not found and no database manager provided"
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "metrics": None
                }

        try:
            # Отримуємо навчену модель
            model_info = self.models[model_key]
            fit_result = model_info.get("fit_result")

            if fit_result is None:
                error_msg = f"Model {model_key} has no fit result"
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "metrics": None
                }

            # Отримуємо прогноз для тестового періоду
            steps = len(test_data)
            self.logger.info(f"Generating in-sample forecasts for {steps} test points")

            # Визначаємо тип моделі
            model_type = model_info.get("metadata", {}).get("model_type", "ARIMA")

            # Генеруємо прогноз в залежності від типу моделі
            if model_type == "ARIMA":
                # Для ARIMA використовуємо прямий метод forecast
                forecast = fit_result.forecast(steps=steps)
            elif model_type == "SARIMA":
                # Для SARIMA використовуємо get_forecast
                forecast_result = fit_result.get_forecast(steps=steps)
                forecast = forecast_result.predicted_mean
            else:
                error_msg = f"Unknown model type: {model_type}"
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "metrics": None
                }

            # Приводимо прогноз до формату Series з тим же індексом, що і тестові дані
            try:
                if len(forecast) != len(test_data):
                    # Якщо довжини не збігаються, обрізаємо прогноз
                    self.logger.warning(
                        f"Forecast length ({len(forecast)}) does not match test data length ({len(test_data)})")
                    min_len = min(len(forecast), len(test_data))
                    forecast = forecast[:min_len]
                    test_data = test_data[:min_len]

                # Створюємо Series з тим же індексом, що і тестові дані
                forecast_series = pd.Series(forecast, index=test_data.index)
            except Exception as e:
                self.logger.warning(f"Error creating forecast series: {str(e)}")
                # Спробуємо створити Series без індексу
                forecast_series = pd.Series(forecast)

            # Обчислюємо метрики
            self.logger.info("Computing evaluation metrics")

            # Перетворюємо дані в numpy масиви для розрахунків
            y_true = test_data.values
            y_pred = forecast_series.values

            # Обчислюємо основні метрики
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)

            # Обчислюємо MAPE (Mean Absolute Percentage Error)
            # Вимагає обережності через можливі нульові значення
            try:
                mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
            except Exception as e:
                self.logger.warning(f"Error calculating MAPE: {str(e)}. Using alternative method.")
                # Альтернативний підхід для запобігання ділення на нуль
                epsilon = np.finfo(float).eps  # Дуже мале число
                mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

            # Додаткові метрики: коефіцієнт детермінації R²
            try:
                from sklearn.metrics import r2_score
                r2 = r2_score(y_true, y_pred)
            except Exception as e:
                self.logger.warning(f"Error calculating R2: {str(e)}")
                r2 = None

            # Збираємо метрики в словник
            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "r2": float(r2) if r2 is not None else None,
                "sample_size": len(y_true),
                "evaluation_date": datetime.now().isoformat()
            }

            # Зберігаємо метрики в БД, якщо є підключення
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Saving metrics for model {model_key} to database")
                    self.db_manager.save_model_metrics(model_key, metrics)
                    self.logger.info(f"Metrics for model {model_key} saved successfully")
                except Exception as e:
                    self.logger.error(f"Error saving metrics to database: {str(e)}")

            # Формуємо результат
            result = {
                "status": "success",
                "message": "Model evaluation completed successfully",
                "metrics": metrics,
                # Додаємо графічне представлення для можливого відображення
                "visual_data": {
                    "actuals": test_data.tolist(),
                    "predictions": forecast_series.tolist(),
                    "dates": [str(idx) for idx in test_data.index]
                }
            }

            self.logger.info(f"Evaluation of model {model_key} completed successfully")
            return result

        except Exception as e:
            error_msg = f"Error during model evaluation: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "metrics": None
            }

    def save_model(self, model_key: str, path: str) -> bool:

        self.logger.info(f"Starting to save model {model_key} to {path}")

        # Перевіряємо наявність моделі в пам'яті
        if model_key not in self.models:
            # Якщо моделі немає в пам'яті, спробуємо завантажити її з БД
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Model {model_key} not found in memory, trying to load from database")
                    loaded = self.db_manager.load_complete_model(model_key)
                    if loaded:
                        self.logger.info(f"Model {model_key} successfully loaded from database")
                    else:
                        error_msg = f"Failed to load model {model_key} from database"
                        self.logger.error(error_msg)
                        return False
                except Exception as e:
                    error_msg = f"Error loading model {model_key} from database: {str(e)}"
                    self.logger.error(error_msg)
                    return False
            else:
                error_msg = f"Model {model_key} not found and no database manager provided"
                self.logger.error(error_msg)
                return False

        try:
            # Отримуємо модель
            model_info = self.models[model_key]

            # Перевіряємо наявність необхідних компонентів
            if "fit_result" not in model_info:
                error_msg = f"Model {model_key} has no fit result"
                self.logger.error(error_msg)
                return False

            # Створюємо директорію, якщо вона не існує
            import os
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"Created directory: {directory}")

            # Зберігаємо модель
            import pickle

            # Створюємо словник з усіма необхідними даними моделі
            model_data = {
                "model_key": model_key,
                "fit_result": model_info["fit_result"],
                "metadata": model_info.get("metadata", {}),
                "parameters": model_info.get("parameters", {}),
                "stats": model_info.get("stats", {}),
                "transformations": self.transformations.get(model_key, {})
            }

            # Зберігаємо в файл
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Model {model_key} successfully saved to {path}")
            return True

        except Exception as e:
            error_msg = f"Error saving model to disk: {str(e)}"
            self.logger.error(error_msg)
            return False

    def load_model(self, model_key: str, path: str) -> bool:

        self.logger.info(f"Loading model from path: {path}")

        try:
            import pickle
            import os

            # Перевірка існування файлу
            if not os.path.exists(path):
                self.logger.error(f"Model file not found at: {path}")
                return False

            # Завантаження моделі з файлу
            with open(path, 'rb') as file:
                loaded_data = pickle.load(file)

            # Перевірка структури завантажених даних
            required_keys = ["model", "fit_result", "metadata", "parameters", "stats"]
            if not all(key in loaded_data for key in required_keys):
                self.logger.error("Loaded model data has incorrect structure")
                return False

            # Записуємо модель у внутрішній словник
            self.models[model_key] = loaded_data

            # Оновлюємо ключ моделі в метаданих, якщо він відрізняється
            self.models[model_key]["metadata"]["model_key"] = model_key

            # Зберігаємо модель в БД, якщо доступний менеджер БД
            if self.db_manager is not None:
                try:
                    # Створюємо бінарне представлення моделі
                    model_binary = pickle.dumps(loaded_data["fit_result"])

                    # Зберігаємо всі компоненти моделі
                    self.db_manager.save_complete_model(
                        model_key=model_key,
                        model_metadata=loaded_data["metadata"],
                        model_parameters=loaded_data["parameters"],
                        model_metrics=loaded_data.get("metrics", {}),  # Може бути відсутнім
                        model_binary=model_binary,
                        data_transformations=loaded_data.get("transformations", {})  # Може бути відсутнім
                    )
                    self.logger.info(f"Model {model_key} saved to database after loading from file")
                except Exception as db_error:
                    self.logger.error(f"Error saving loaded model to database: {str(db_error)}")

            self.logger.info(f"Model {model_key} successfully loaded from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model from file: {str(e)}")
            return False

    def transform_data(self, data: pd.Series, method: str = 'log') -> Union[pd.Series, Tuple[pd.Series, float]]:

        self.logger.info(f"Applying {method} transformation to data")

        # Перевірка наявності нульових та від'ємних значень для певних трансформацій
        if method in ['log', 'boxcox']:
            min_value = data.min()
            if min_value <= 0:
                self.logger.warning(f"Data contains non-positive values, adding offset for {method} transformation")
                offset = abs(min_value) + 1
                data = data + offset

                # Зберігаємо інформацію про зсув для подальшої зворотної трансформації
                self.transformations[method] = {'offset': offset}
            else:
                self.transformations[method] = {'offset': 0}

        if method == 'none':
            # Без трансформації
            return data

        elif method == 'log':
            # Логарифмічна трансформація
            transformed_data = np.log(data)
            return transformed_data

        elif method == 'sqrt':
            # Квадратний корінь
            transformed_data = np.sqrt(data)
            return transformed_data

        elif method == 'boxcox':
            # Трансформація Бокса-Кокса
            transformed_data, lambda_param = stats.boxcox(data)

            # Зберігаємо lambda параметр для зворотної трансформації
            if method in self.transformations:
                self.transformations[method]['lambda'] = lambda_param
            else:
                self.transformations[method] = {'lambda': lambda_param, 'offset': 0}

            self.logger.info(f"BoxCox transformation applied with lambda = {lambda_param}")
            return transformed_data, lambda_param

        elif method == 'yeo-johnson':
            # Трансформація Йео-Джонсона (працює з від'ємними значеннями)
            from sklearn.preprocessing import PowerTransformer

            # Підготовка даних для трансформації
            data_reshaped = data.values.reshape(-1, 1)

            # Застосування трансформації
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            transformed_data_array = pt.fit_transform(data_reshaped)
            transformed_data = pd.Series(transformed_data_array.flatten(), index=data.index)

            # Зберігаємо параметри трансформації
            lambda_param = pt.lambdas_[0]
            self.transformations['yeo-johnson'] = {'lambda': lambda_param, 'transformer': pt}

            self.logger.info(f"Yeo-Johnson transformation applied with lambda = {lambda_param}")
            return transformed_data, lambda_param

        else:
            error_msg = f"Unknown transformation method: {method}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def inverse_transform(self, data: pd.Series, method: str = 'log', lambda_param: float = None) -> pd.Series:

        self.logger.info(f"Applying inverse {method} transformation")

        if method == 'none':
            # Без трансформації
            return data

        # Отримуємо збережені параметри трансформації
        transform_params = self.transformations.get(method, {})
        offset = transform_params.get('offset', 0)

        if method == 'log':
            # Зворотна логарифмічна трансформація
            inverse_data = np.exp(data)

            # Відновлюємо зсув, якщо він був застосований
            if offset > 0:
                inverse_data = inverse_data - offset

            return inverse_data

        elif method == 'sqrt':
            # Зворотна трансформація квадратного кореня
            inverse_data = data ** 2

            # Відновлюємо зсув, якщо він був застосований
            if offset > 0:
                inverse_data = inverse_data - offset

            return inverse_data

        elif method == 'boxcox':
            # Зворотна трансформація Бокса-Кокса

            # Якщо lambda_param не вказано, намагаємося використати збережений
            if lambda_param is None:
                lambda_param = transform_params.get('lambda')

                if lambda_param is None:
                    error_msg = "lambda parameter for BoxCox inverse transformation is not provided or saved"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            # Застосовуємо зворотну трансформацію
            inverse_data = stats.inv_boxcox(data, lambda_param)

            # Відновлюємо зсув, якщо він був застосований
            if offset > 0:
                inverse_data = inverse_data - offset

            return inverse_data

        elif method == 'yeo-johnson':
            # Зворотна трансформація Йео-Джонсона

            # Отримуємо збережений трансформер
            transformer = transform_params.get('transformer')

            if transformer is None:
                error_msg = "Transformer object for Yeo-Johnson inverse transformation is not saved"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Підготовка даних
            data_reshaped = data.values.reshape(-1, 1)

            # Застосування зворотної трансформації
            inverse_data_array = transformer.inverse_transform(data_reshaped)
            inverse_data = pd.Series(inverse_data_array.flatten(), index=data.index)

            return inverse_data

        else:
            error_msg = f"Unknown transformation method: {method}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def detect_seasonality(self, data: pd.Series) -> Dict:

        self.logger.info("Початок аналізу сезонності")

        # Перевірка на наявність пропущених значень
        if data.isnull().any():
            self.logger.warning("Дані містять пропущені значення (NaN). Видаляємо їх перед аналізом.")
            data = data.dropna()

        # Перевірка достатньої кількості точок даних
        if len(data) < 24:
            self.logger.error("Недостатньо даних для надійного виявлення сезонності (мінімум 24 точки)")
            return {
                "status": "error",
                "message": "Недостатньо даних для надійного виявлення сезонності",
                "has_seasonality": False,
                "seasonal_periods": [],
                "details": {}
            }

        # Ініціалізація результату
        result = {
            "status": "success",
            "message": "Аналіз сезонності завершено",
            "has_seasonality": False,
            "seasonal_periods": [],
            "details": {}
        }

        try:


            # 2. Автокореляційний аналіз
            from statsmodels.tsa.stattools import acf, pacf

            # Обчислюємо максимальну кількість лагів (до половини довжини ряду або 50)
            max_lags = min(len(data) // 2, 50)

            # Обчислюємо ACF (автокореляційна функція)
            acf_values = acf(data, nlags=max_lags, fft=True)

            # Обчислюємо PACF (часткова автокореляційна функція)
            pacf_values = pacf(data, nlags=max_lags)

            # 3. Пошук значимих лагів у ACF
            # Поріг значимості (зазвичай 1.96/sqrt(n) для 95% довірчого інтервалу)
            significance_threshold = 1.96 / np.sqrt(len(data))

            # Знаходимо значимі лаги (з автокореляцією вище порогу)
            significant_lags = [lag for lag in range(2, len(acf_values))
                                if abs(acf_values[lag]) > significance_threshold]

            result["details"]["acf_analysis"] = {
                "significant_lags": significant_lags,
                "significance_threshold": significance_threshold
            }

            # 4. Типові сезонні періоди для фінансових даних
            typical_periods = [7, 14, 30, 90, 365]  # Тижневий, двотижневий, місячний, квартальний, річний

            # Визначаємо потенційні сезонні періоди зі значимих лагів
            potential_seasonal_periods = []

            # Шукаємо локальні піки в ACF як потенційні сезонні періоди
            for lag in range(2, len(acf_values) - 1):
                if (acf_values[lag] > acf_values[lag - 1] and
                        acf_values[lag] > acf_values[lag + 1] and
                        abs(acf_values[lag]) > significance_threshold):
                    potential_seasonal_periods.append({
                        "lag": lag,
                        "acf_value": acf_values[lag],
                        "strength": abs(acf_values[lag]) / abs(acf_values[0])  # Відносна сила
                    })

            # Сортуємо за силою кореляції
            potential_seasonal_periods.sort(key=lambda x: x["strength"], reverse=True)

            # 5. Декомпозиція часового ряду
            try:
                # Визначаємо період для декомпозиції
                if len(potential_seasonal_periods) > 0:
                    decomposition_period = potential_seasonal_periods[0]["lag"]
                else:
                    # Використовуємо найбільш ймовірний період з типових
                    for period in typical_periods:
                        if period < len(data) // 2:
                            decomposition_period = period
                            break
                    else:
                        decomposition_period = min(len(data) // 4, 7)  # Резервний варіант

                # Перевіряємо, що період більше 1 і підходить для декомпозиції
                if decomposition_period < 2:
                    decomposition_period = 2

                # Сезонна декомпозиція
                decomposition = seasonal_decompose(
                    data,
                    model='additive',
                    period=decomposition_period,
                    extrapolate_trend='freq'
                )

                # Витягуємо сезонний компонент
                seasonal_component = decomposition.seasonal

                # Розраховуємо силу сезонності як відношення дисперсії сезонного компоненту до загальної дисперсії
                seasonal_strength = np.var(seasonal_component) / np.var(data)

                result["details"]["decomposition"] = {
                    "period_used": decomposition_period,
                    "seasonal_strength": seasonal_strength,
                    "model": "additive"
                }

                # Якщо сила сезонності значна, вважаємо що ряд має сезонність
                if seasonal_strength > 0.1:  # Поріг 10%
                    result["has_seasonality"] = True

            except Exception as e:
                self.logger.warning(f"Помилка під час сезонної декомпозиції: {str(e)}")
                result["details"]["decomposition"] = {
                    "error": str(e)
                }

            # 6. Тест на наявність сезонності за допомогою спектрального аналізу
            try:
                from scipy import signal

                # Створюємо рівномірні часові точки для спектрального аналізу
                if not isinstance(data.index, pd.DatetimeIndex):
                    t = np.arange(len(data))
                else:
                    # Для часового індексу конвертуємо в дні від початку
                    t = (data.index - data.index[0]).total_seconds() / (24 * 3600)

                # Розраховуємо спектр за допомогою періодограми
                freqs, spectrum = signal.periodogram(data.values, fs=1.0)

                # Виключаємо нульову частоту (постійний компонент)
                freqs = freqs[1:]
                spectrum = spectrum[1:]

                # Знаходимо піки в спектрі
                peaks, _ = signal.find_peaks(spectrum, height=np.max(spectrum) / 10)

                if len(peaks) > 0:
                    # Конвертуємо частоти в періоди (періоди = 1/частота)
                    peak_periods = [round(1.0 / freqs[p]) for p in peaks if freqs[p] > 0]

                    # Відфільтровуємо занадто великі або малі періоди
                    filtered_periods = [p for p in peak_periods if 2 <= p <= len(data) // 3]

                    result["details"]["spectral_analysis"] = {
                        "detected_periods": filtered_periods,
                        "peak_count": len(peaks)
                    }

                    # Доповнюємо список можливих сезонних періодів
                    for period in filtered_periods:
                        if period not in [p["lag"] for p in potential_seasonal_periods]:
                            potential_seasonal_periods.append({
                                "lag": period,
                                "source": "spectral",
                                "strength": 0.8  # Приблизна оцінка сили
                            })

                else:
                    result["details"]["spectral_analysis"] = {
                        "detected_periods": [],
                        "peak_count": 0
                    }

            except Exception as e:
                self.logger.warning(f"Помилка під час спектрального аналізу: {str(e)}")
                result["details"]["spectral_analysis"] = {
                    "error": str(e)
                }

            # 7. Формуємо підсумковий список сезонних періодів з оцінкою впевненості
            seasonal_periods = []

            # Сумуємо всі знайдені потенційні періоди
            for period in potential_seasonal_periods:
                lag = period["lag"]

                # Розраховуємо впевненість на основі сили та інших факторів
                confidence = period.get("strength", 0.5)

                # Підвищуємо впевненість, якщо період відповідає типовим
                if any(abs(lag - typical) / typical < 0.1 for typical in typical_periods):
                    confidence += 0.2

                # Підвищуємо впевненість, якщо період підтверджено кількома методами
                sources = []
                if "acf_value" in period:
                    sources.append("acf")
                if period.get("source") == "spectral":
                    sources.append("spectral")
                if result.get("has_seasonality") and abs(
                        lag - result["details"]["decomposition"].get("period_used", 0)) < 2:
                    sources.append("decomposition")

                confidence = min(confidence + 0.1 * (len(sources) - 1), 1.0)  # Максимум 1.0

                seasonal_periods.append({
                    "period": lag,
                    "confidence": confidence,
                    "sources": sources
                })

            # Сортуємо за впевненістю
            seasonal_periods.sort(key=lambda x: x["confidence"], reverse=True)

            # Видаляємо дублікати та близькі періоди (залишаємо більш впевнений)
            filtered_periods = []
            for period in seasonal_periods:
                if not any(abs(period["period"] - existing["period"]) / existing["period"] < 0.1
                           for existing in filtered_periods):
                    filtered_periods.append(period)

            # Записуємо в результат
            result["seasonal_periods"] = filtered_periods

            # Визначаємо наявність сезонності на основі всіх факторів
            if len(filtered_periods) > 0 and filtered_periods[0]["confidence"] > 0.7:
                result["has_seasonality"] = True
                result["primary_period"] = filtered_periods[0]["period"]

            self.logger.info(f"Аналіз сезонності завершено: {result['has_seasonality']}")
            if result["has_seasonality"]:
                self.logger.info(f"Основний сезонний період: {result.get('primary_period')}")

            return result

        except Exception as e:
            self.logger.error(f"Помилка під час аналізу сезонності: {str(e)}")
            return {
                "status": "error",
                "message": f"Помилка під час аналізу сезонності: {str(e)}",
                "has_seasonality": False,
                "seasonal_periods": [],
                "details": {}
            }

    def rolling_window_validation(self, data: pd.Series, model_type: str = 'arima',
                                  order: Tuple = None, seasonal_order: Tuple = None,
                                  window_size: int = 100, step: int = 20,
                                  forecast_horizon: int = 10) -> Dict:

        self.logger.info(
            f"Початок ковзаючої валідації з вікном={window_size}, кроком={step}, горизонтом={forecast_horizon}")

        # Перевірка вхідних даних
        if data.isnull().any():
            self.logger.warning("Дані містять пропущені значення. Видаляємо їх перед валідацією.")
            data = data.dropna()

        if len(data) < window_size + forecast_horizon:
            error_msg = f"Недостатньо даних для валідації. Потрібно мінімум {window_size + forecast_horizon}, отримано {len(data)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "iterations": 0,
                "metrics": {},
                "forecasts": {}
            }

        # Перевірка параметрів
        if model_type not in ['arima', 'sarima']:
            self.logger.error(f"Невірний тип моделі: {model_type}. Має бути 'arima' або 'sarima'")
            return {
                "status": "error",
                "message": f"Невірний тип моделі: {model_type}. Має бути 'arima' або 'sarima'",
                "iterations": 0,
                "metrics": {},
                "forecasts": {}
            }

        # Перевірка та встановлення параметрів моделі
        if model_type == 'arima' and order is None:
            self.logger.warning("Не вказано параметри для ARIMA моделі. Використовуємо за замовчуванням (1,1,1)")
            order = (1, 1, 1)

        if model_type == 'sarima' and (order is None or seasonal_order is None):
            self.logger.warning("Не вказано параметри для SARIMA моделі. Використовуємо значення за замовчуванням")
            if order is None:
                order = (1, 1, 1)
            if seasonal_order is None:
                # Визначаємо сезонний період
                seasonal_result = self.detect_seasonality(data)
                if seasonal_result["has_seasonality"] and "primary_period" in seasonal_result:
                    seasonal_period = seasonal_result["primary_period"]
                else:
                    seasonal_period = 7  # Тижнева сезонність за замовчуванням

                seasonal_order = (1, 0, 1, seasonal_period)
                self.logger.info(f"Використовуємо сезонні параметри за замовчуванням: {seasonal_order}")

        # Визначаємо кількість ітерацій
        total_iterations = (len(data) - window_size - forecast_horizon) // step + 1

        if total_iterations <= 0:
            self.logger.error("Недостатньо даних для жодної ітерації валідації")
            return {
                "status": "error",
                "message": "Недостатньо даних для жодної ітерації валідації",
                "iterations": 0,
                "metrics": {},
                "forecasts": {}
            }

        self.logger.info(f"Буде виконано {total_iterations} ітерацій валідації")

        # Зберігання результатів
        results = {
            "status": "success",
            "message": "Ковзаюча валідація завершена",
            "iterations": total_iterations,
            "metrics": {
                "mse": [],
                "rmse": [],
                "mae": [],
                "mape": []
            },
            "forecasts": {}
        }

        # Визначаємо, чи використовується DatetimeIndex
        is_datetime_index = isinstance(data.index, pd.DatetimeIndex)

        # Запускаємо процес валідації
        for i in range(total_iterations):
            iteration_start_time = datetime.now()

            # Визначаємо індекси для поточної ітерації
            start_idx = i * step
            end_train_idx = start_idx + window_size
            end_test_idx = end_train_idx + forecast_horizon

            # Якщо залишилося недостатньо даних для прогнозу
            if end_test_idx > len(data):
                break

            # Отримуємо навчальні дані
            train_data = data.iloc[start_idx:end_train_idx].copy()

            # Отримуємо тестові дані для порівняння
            test_data = data.iloc[end_train_idx:end_test_idx].copy()

            self.logger.info(
                f"Ітерація {i + 1}/{total_iterations}: Навчання на {len(train_data)} точках, тестування на {len(test_data)} точках")

            # Навчаємо модель
            try:
                if model_type == 'arima':
                    # Навчаємо ARIMA модель
                    model = ARIMA(train_data, order=order)
                    fit_result = model.fit()

                    # Виконуємо прогноз
                    forecast = fit_result.forecast(steps=forecast_horizon)

                elif model_type == 'sarima':
                    # Навчаємо SARIMA модель
                    model = SARIMAX(
                        train_data,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    fit_result = model.fit(disp=False)

                    # Виконуємо прогноз
                    forecast = fit_result.forecast(steps=forecast_horizon)

                # Перевіряємо прогноз на пропуски та нескінченні значення
                if np.isnan(forecast).any() or np.isinf(forecast).any():
                    self.logger.warning(
                        f"Ітерація {i + 1}: Прогноз містить NaN або нескінченні значення. Пропускаємо ітерацію.")
                    continue

                # Обчислюємо метрики
                # Обробляємо випадок, коли тестові та прогнозні дані мають різні індекси
                if isinstance(forecast, pd.Series) and not forecast.index.equals(test_data.index):
                    forecast_values = forecast.values
                else:
                    forecast_values = forecast

                # Розрахунок метрик
                mse = mean_squared_error(test_data, forecast_values)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_data, forecast_values)

                # MAPE (тільки для ненульових значень)
                if (np.abs(test_data) > 1e-10).all():
                    mape = np.mean(np.abs((test_data - forecast_values) / test_data)) * 100
                else:
                    # Альтернативний розрахунок для випадків з нульовими значеннями
                    mape = np.mean(np.abs((test_data - forecast_values) / (test_data + 1e-10))) * 100
                    self.logger.warning(
                        f"Ітерація {i + 1}: Деякі тестові значення близькі до нуля. MAPE може бути ненадійним.")

                # Зберігаємо метрики
                results["metrics"]["mse"].append(mse)
                results["metrics"]["rmse"].append(rmse)
                results["metrics"]["mae"].append(mae)
                results["metrics"]["mape"].append(mape)

                # Зберігаємо прогноз та реальні значення
                if is_datetime_index:
                    # Зберігаємо дати у ISO форматі для JSON-сумісності
                    iteration_key = f"iter_{i + 1}"
                    results["forecasts"][iteration_key] = {
                        "train_period": {
                            "start": train_data.index[0].isoformat(),
                            "end": train_data.index[-1].isoformat()
                        },
                        "test_period": {
                            "start": test_data.index[0].isoformat(),
                            "end": test_data.index[-1].isoformat()
                        },
                        "metrics": {
                            "mse": mse,
                            "rmse": rmse,
                            "mae": mae,
                            "mape": mape
                        },
                        # Конвертуємо індекси та значення у списки для JSON
                        "actual": list(zip(
                            [idx.isoformat() for idx in test_data.index],
                            test_data.values.tolist()
                        )),
                        "forecast": list(zip(
                            [idx.isoformat() for idx in test_data.index],
                            forecast_values.tolist() if hasattr(forecast_values, 'tolist') else list(forecast_values)
                        ))
                    }
                else:
                    # Для звичайних індексів
                    iteration_key = f"iter_{i + 1}"
                    results["forecasts"][iteration_key] = {
                        "train_indices": {
                            "start": int(start_idx),
                            "end": int(end_train_idx - 1)
                        },
                        "test_indices": {
                            "start": int(end_train_idx),
                            "end": int(end_test_idx - 1)
                        },
                        "metrics": {
                            "mse": float(mse),
                            "rmse": float(rmse),
                            "mae": float(mae),
                            "mape": float(mape)
                        },
                        "actual": test_data.values.tolist(),
                        "forecast": forecast_values.tolist() if hasattr(forecast_values, 'tolist') else list(
                            forecast_values)
                    }

                # Зберігаємо в БД, якщо менеджер БД доступний
                if self.db_manager is not None:
                    try:
                        # Створюємо унікальний ключ для ітерації
                        validation_key = f"validation_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i + 1}"

                        # Створюємо метрики у форматі для БД
                        db_metrics = {
                            "model_type": model_type,
                            "order": str(order),
                            "seasonal_order": str(seasonal_order) if model_type == 'sarima' else None,
                            "window_size": window_size,
                            "forecast_horizon": forecast_horizon,
                            "iteration": i + 1,
                            "train_start": train_data.index[0] if is_datetime_index else int(start_idx),
                            "train_end": train_data.index[-1] if is_datetime_index else int(end_train_idx - 1),
                            "test_start": test_data.index[0] if is_datetime_index else int(end_train_idx),
                            "test_end": test_data.index[-1] if is_datetime_index else int(end_test_idx - 1),
                            "mse": float(mse),
                            "rmse": float(rmse),
                            "mae": float(mae),
                            "mape": float(mape),
                            "timestamp": datetime.now()
                        }

                        # Зберігаємо метрики в БД
                        self.db_manager.save_model_metrics(validation_key, db_metrics)
                        self.logger.info(f"Метрики ітерації {i + 1} збережено в БД з ключем {validation_key}")

                    except Exception as db_error:
                        self.logger.error(f"Помилка збереження метрик валідації в БД: {str(db_error)}")

                self.logger.info(
                    f"Ітерація {i + 1} завершена за {datetime.now() - iteration_start_time}. MSE: {mse:.4f}, RMSE: {rmse:.4f}")

            except Exception as e:
                self.logger.error(f"Помилка під час ітерації {i + 1}: {str(e)}")
                results["forecasts"][f"iter_{i + 1}_error"] = {
                    "error": str(e),
                    "train_indices": {
                        "start": int(start_idx),
                        "end": int(end_train_idx - 1)
                    }
                }

        # Розраховуємо підсумкові метрики (якщо були успішні ітерації)
        if results["metrics"]["mse"]:
            results["aggregated_metrics"] = {
                "mean_mse": np.mean(results["metrics"]["mse"]),
                "mean_rmse": np.mean(results["metrics"]["rmse"]),
                "mean_mae": np.mean(results["metrics"]["mae"]),
                "mean_mape": np.mean(results["metrics"]["mape"]),
                "std_mse": np.std(results["metrics"]["mse"]),
                "std_rmse": np.std(results["metrics"]["rmse"]),
                "std_mae": np.std(results["metrics"]["mae"]),
                "std_mape": np.std(results["metrics"]["mape"]),
                "min_mse": np.min(results["metrics"]["mse"]),
                "max_mse": np.max(results["metrics"]["mse"]),
                "successful_iterations": len(results["metrics"]["mse"])
            }

            self.logger.info(
                f"Ковзаюча валідація завершена. Середнє RMSE: {results['aggregated_metrics']['mean_rmse']:.4f}")
        else:
            results["status"] = "warning"
            results["message"] = "Ковзаюча валідація завершена, але немає успішних ітерацій"
            self.logger.warning("Ковзаюча валідація завершена, але немає успішних ітерацій")

        return results

    def residual_analysis(self, model_key: str, data: pd.Series = None) -> Dict:

        self.logger.info(f"Starting residual analysis for model {model_key}")

        # Перевірка наявності моделі
        if model_key not in self.models:
            # Спробувати завантажити модель з БД
            if self.db_manager is not None:
                try:
                    model_loaded = self.db_manager.load_complete_model(model_key)
                    if not model_loaded:
                        error_msg = f"Model {model_key} not found in database"
                        self.logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                except Exception as e:
                    error_msg = f"Error loading model {model_key} from database: {str(e)}"
                    self.logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
            else:
                error_msg = f"Model {model_key} not found and no database manager available"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

        try:
            # Отримання моделі
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

            # Використання даних моделі, якщо дані не надані
            if data is None:
                # В залежності від моделі, можуть використовуватися різні підходи до отримання даних
                try:
                    # Для моделей ARIMA/SARIMA з statsmodels
                    data = fit_result.model.endog
                    self.logger.info(f"Using original model data for residual analysis, length: {len(data)}")
                except Exception as e:
                    error_msg = f"Error accessing model data: {str(e)}"
                    self.logger.error(error_msg)
                    return {"status": "error", "message": error_msg}

            # Отримання залишків
            residuals = fit_result.resid

            # Базова статистика залишків
            residuals_stats = {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "min": float(residuals.min()),
                "max": float(residuals.max()),
                "median": float(np.median(residuals))
            }

            # Тест Льюнга-Бокса на автокореляцію залишків
            from statsmodels.stats.diagnostic import acorr_ljungbox
            max_lag = min(10, len(residuals) // 5)  # Не більше 1/5 від довжини ряду
            lb_results = acorr_ljungbox(residuals, lags=max_lag)

            # Тест на нормальність розподілу залишків (Jarque-Bera)
            from scipy import stats
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)

            # Тест на гетероскедастичність (Breusch-Pagan)
            from statsmodels.stats.diagnostic import het_breuschpagan
            try:
                # Створюємо штучний регресор - порядковий номер спостереження
                X = np.arange(1, len(residuals) + 1).reshape(-1, 1)
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
            except Exception as e:
                self.logger.warning(f"Error during Breusch-Pagan test: {str(e)}")
                bp_stat, bp_pvalue = None, None

            # Автокореляційна функція (ACF) та часткова автокореляційна функція (PACF) залишків
            from statsmodels.tsa.stattools import acf, pacf
            acf_values = acf(residuals, nlags=max_lag, fft=True)
            pacf_values = pacf(residuals, nlags=max_lag)

            # Визначення значущості автокореляції
            # 95% довірчий інтервал для ACF (приблизно ±1.96/sqrt(N))
            conf_interval = 1.96 / np.sqrt(len(residuals))

            significant_acf = [i for i, v in enumerate(acf_values) if i > 0 and abs(v) > conf_interval]
            significant_pacf = [i for i, v in enumerate(pacf_values) if i > 0 and abs(v) > conf_interval]

            # Оцінка білого шуму (якщо залишки - білий шум, модель добре підібрана)
            is_white_noise = len(significant_acf) <= max_lag * 0.05  # Не більше 5% значущих лагів

            # Формування результатів аналізу
            analysis_results = {
                "status": "success",
                "model_key": model_key,
                "residuals_statistics": residuals_stats,
                "normality_test": {
                    "jarque_bera_statistic": float(jb_stat),
                    "jarque_bera_pvalue": float(jb_pvalue),
                    "is_normal": jb_pvalue > 0.05
                },
                "autocorrelation_test": {
                    "ljung_box_statistic": [float(stat) for stat in lb_results.lb_stat],
                    "ljung_box_pvalue": [float(pval) for pval in lb_results.lb_pvalue],
                    "has_autocorrelation": any(pval < 0.05 for pval in lb_results.lb_pvalue)
                },
                "heteroscedasticity_test": {
                    "breusch_pagan_statistic": float(bp_stat) if bp_stat is not None else None,
                    "breusch_pagan_pvalue": float(bp_pvalue) if bp_pvalue is not None else None,
                    "has_heteroscedasticity": bp_pvalue < 0.05 if bp_pvalue is not None else None
                },
                "acf_analysis": {
                    "values": [float(val) for val in acf_values],
                    "significant_lags": significant_acf,
                    "confidence_interval": float(conf_interval)
                },
                "pacf_analysis": {
                    "values": [float(val) for val in pacf_values],
                    "significant_lags": significant_pacf,
                    "confidence_interval": float(conf_interval)
                },
                "white_noise_assessment": {
                    "is_white_noise": is_white_noise,
                    "explanation": "Residuals appear to be white noise" if is_white_noise
                    else "Residuals show patterns, model may be improved"
                },
                "timestamp": datetime.now().isoformat()
            }

            # Зберегти результати аналізу в БД, якщо є DB manager
            if self.db_manager is not None:
                try:

                    self.db_manager.save_residual_analysis(model_key, analysis_results)
                    self.logger.info(f"Residual analysis for model {model_key} saved to database")
                except Exception as e:
                    self.logger.warning(f"Error saving residual analysis to database: {str(e)}")

            self.logger.info(f"Residual analysis for model {model_key} completed successfully")
            return analysis_results

        except Exception as e:
            error_msg = f"Error during residual analysis: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def forecast_with_intervals(self, model_key: str, steps: int = 24,
                                alpha: float = 0.05) -> Dict:

        self.logger.info(f"Starting forecast with intervals for model {model_key}, steps={steps}, alpha={alpha}")

        # Перевірка значення alpha
        if alpha <= 0 or alpha >= 1:
            error_msg = f"Invalid alpha value ({alpha}). Must be between 0 and 1."
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Перевірка наявності моделі
        if model_key not in self.models:
            # Спробувати завантажити модель з БД
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
            # Отримання моделі
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

            # Отримання інформації про трансформації даних
            transformations = None
            if self.db_manager is not None:
                try:
                    transformations = self.db_manager.get_data_transformations(model_key)
                    if transformations:
                        self.logger.info(f"Found data transformations for model {model_key}")
                except Exception as e:
                    self.logger.warning(f"Error getting data transformations: {str(e)}")

            # Виконання прогнозування з довірчими інтервалами
            try:
                # Прогнозування з довірчими інтервалами
                forecast_result = fit_result.get_forecast(steps=steps)

                # Отримання прогнозних значень та інтервалів
                predicted_mean = forecast_result.predicted_mean
                confidence_intervals = forecast_result.conf_int(alpha=alpha)

                # Створення часових індексів для прогнозу
                # Визначаємо частоту даних з оригінальної моделі
                if hasattr(fit_result.model.data, 'dates') and fit_result.model.data.dates is not None:
                    original_index = fit_result.model.data.dates
                    # Визначаємо частоту
                    if isinstance(original_index, pd.DatetimeIndex):
                        freq = pd.infer_freq(original_index)
                        if freq is None:
                            # Спробуємо вгадати частоту на основі різниць
                            if len(original_index) > 1:
                                avg_diff = (original_index[-1] - original_index[0]) / (len(original_index) - 1)
                                if avg_diff.days >= 1:
                                    freq = f"{avg_diff.days}D"
                                else:
                                    hours = avg_diff.seconds // 3600
                                    if hours >= 1:
                                        freq = f"{hours}H"
                                    else:
                                        minutes = (avg_diff.seconds % 3600) // 60
                                        if minutes >= 1:
                                            freq = f"{minutes}min"
                                        else:
                                            freq = f"{avg_diff.seconds % 60}S"

                        # Створення нових індексів для прогнозу
                        last_date = original_index[-1]
                        forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
                    else:
                        # Якщо індекс не datetime, використовуємо числові індекси
                        last_idx = len(original_index)
                        forecast_index = pd.RangeIndex(start=last_idx, stop=last_idx + steps)
                else:
                    # Якщо немає інформації про дати, використовуємо числові індекси
                    # Спробуємо вгадати останній індекс
                    if hasattr(fit_result.model, 'endog') and hasattr(fit_result.model.endog, 'shape'):
                        last_idx = fit_result.model.endog.shape[0]
                        forecast_index = pd.RangeIndex(start=last_idx, stop=last_idx + steps)
                    else:
                        forecast_index = pd.RangeIndex(start=0, stop=steps)

                # Створення Series для прогнозу та інтервалів
                forecast_series = pd.Series(predicted_mean, index=forecast_index)
                lower_bound = pd.Series(confidence_intervals.iloc[:, 0].values, index=forecast_index)
                upper_bound = pd.Series(confidence_intervals.iloc[:, 1].values, index=forecast_index)

                # Зворотна трансформація, якщо потрібно
                if transformations:
                    try:
                        transform_method = transformations.get("method")
                        transform_param = transformations.get("lambda_param")

                        if transform_method:
                            self.logger.info(f"Applying inverse transformation: {transform_method}")
                            forecast_series = self.inverse_transform(forecast_series, method=transform_method,
                                                                     lambda_param=transform_param)
                            lower_bound = self.inverse_transform(lower_bound, method=transform_method,
                                                                 lambda_param=transform_param)
                            upper_bound = self.inverse_transform(upper_bound, method=transform_method,
                                                                 lambda_param=transform_param)
                    except Exception as e:
                        self.logger.warning(f"Error during inverse transformation: {str(e)}")

                # Формування результатів прогнозу
                forecast_data = {
                    "forecast": forecast_series.tolist(),
                    "lower_bound": lower_bound.tolist(),
                    "upper_bound": upper_bound.tolist(),
                    "indices": [str(idx) for idx in forecast_index],
                    "confidence_level": 1.0 - alpha
                }

                # Збереження результатів прогнозу в БД
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

                # Повний результат
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

    def compare_models(self, model_keys: List[str], test_data: pd.Series) -> Dict:

        self.logger.info(f"Starting comparison of models: {model_keys}")

        if len(model_keys) < 2:
            self.logger.warning("At least two models are required for comparison")
            return {
                "status": "error",
                "message": "At least two models are required for comparison",
                "results": {}
            }

        if test_data.isnull().any():
            self.logger.warning("Test data contains NaN values. Removing them before comparison.")
            test_data = test_data.dropna()

        if len(test_data) == 0:
            self.logger.error("Test data is empty after removing NaN values")
            return {
                "status": "error",
                "message": "Test data is empty",
                "results": {}
            }

        comparison_results = {
            "models": {},
            "best_model": None,
            "metrics": {
                "mse": {},
                "rmse": {},
                "mae": {},
                "mape": {}
            }
        }

        try:
            # Отримати прогнози для кожної моделі і порівняти їх з тестовими даними
            for model_key in model_keys:
                try:
                    # Завантажити модель з БД, якщо потрібно
                    if model_key not in self.models and self.db_manager is not None:
                        self.logger.info(f"Loading model {model_key} from database")
                        loaded_model = self.db_manager.load_complete_model(model_key)
                        if loaded_model:
                            self.models[model_key] = loaded_model
                        else:
                            self.logger.warning(f"Could not load model {model_key} from database")
                            continue

                    if model_key not in self.models:
                        self.logger.error(f"Model {model_key} not found")
                        continue

                    model_info = self.models[model_key]
                    fit_result = model_info["fit_result"]

                    # Отримати прогноз на період тестових даних
                    forecast_start = test_data.index[0]
                    forecast_end = test_data.index[-1]

                    # Генерування прогнозу
                    forecast = fit_result.get_prediction(start=forecast_start, end=forecast_end)
                    pred_mean = forecast.predicted_mean

                    # Вирівняти індекси прогнозу та тестових даних
                    pred_series = pd.Series(pred_mean, index=test_data.index)

                    # Застосування зворотних трансформацій, якщо потрібно
                    if self.db_manager is not None:
                        transformations = self.db_manager.get_data_transformations(model_key)
                        if transformations:
                            for transform in reversed(transformations):
                                if transform.get("method"):
                                    self.logger.info(f"Applying inverse transformation: {transform['method']}")
                                    if transform["method"] == "log":
                                        pred_series = np.exp(pred_series)
                                    elif transform["method"] == "boxcox":
                                        from scipy import special
                                        pred_series = special.inv_boxcox(pred_series, transform.get("lambda", 0))
                                    elif transform["method"] == "sqrt":
                                        pred_series = pred_series ** 2

                    # Обчислення метрик
                    mse = mean_squared_error(test_data, pred_series)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(test_data, pred_series)

                    # Обчислення MAPE з обробкою нульових значень
                    mask = test_data != 0
                    if mask.any():
                        mape = np.mean(np.abs((test_data[mask] - pred_series[mask]) / test_data[mask])) * 100
                    else:
                        mape = np.nan

                    comparison_results["metrics"]["mse"][model_key] = mse
                    comparison_results["metrics"]["rmse"][model_key] = rmse
                    comparison_results["metrics"]["mae"][model_key] = mae
                    comparison_results["metrics"]["mape"][model_key] = mape

                    comparison_results["models"][model_key] = {
                        "forecast": pred_series.to_dict(),
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "mape": mape,
                        "model_type": model_info["metadata"]["model_type"],
                        "parameters": model_info["parameters"]
                    }

                    self.logger.info(
                        f"Evaluated model {model_key}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

                except Exception as e:
                    self.logger.error(f"Error evaluating model {model_key}: {str(e)}")
                    comparison_results["models"][model_key] = {"error": str(e)}

            # Визначення найкращої моделі за MSE (можна змінити критерій)
            valid_models = {k: v for k, v in comparison_results["metrics"]["mse"].items()
                            if isinstance(v, (int, float)) and not np.isnan(v)}

            if valid_models:
                best_model_key = min(valid_models, key=valid_models.get)
                comparison_results["best_model"] = {
                    "key": best_model_key,
                    "metrics": {
                        "mse": comparison_results["metrics"]["mse"][best_model_key],
                        "rmse": comparison_results["metrics"]["rmse"][best_model_key],
                        "mae": comparison_results["metrics"]["mae"][best_model_key],
                        "mape": comparison_results["metrics"]["mape"][best_model_key]
                    }
                }

                self.logger.info(f"Best model: {best_model_key}")

            # Якщо є db_manager, зберегти результати порівняння
            if self.db_manager is not None:
                try:
                    self.db_manager.compare_model_forecasts(
                        model_keys=model_keys,
                        comparison_data={
                            "test_period": {
                                "start": test_data.index[0].isoformat() if isinstance(test_data.index[0],
                                                                                      datetime) else str(
                                    test_data.index[0]),
                                "end": test_data.index[-1].isoformat() if isinstance(test_data.index[-1],
                                                                                     datetime) else str(
                                    test_data.index[-1]),
                            },
                            "metrics": comparison_results["metrics"],
                            "best_model": comparison_results["best_model"][
                                "key"] if "best_model" in comparison_results else None
                        }
                    )
                except Exception as db_error:
                    self.logger.error(f"Error saving comparison results to database: {str(db_error)}")

            return {
                "status": "success",
                "message": "Models compared successfully",
                "results": comparison_results
            }

        except Exception as e:
            self.logger.error(f"Error during model comparison: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during model comparison: {str(e)}",
                "results": {}
            }

    def apply_preprocessing_pipeline(self, data: pd.Series, operations: List[Dict]) -> pd.Series:

        self.logger.info(f"Applying preprocessing pipeline with {len(operations)} operations")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before preprocessing.")
            data = data.dropna()

        if len(data) == 0:
            self.logger.error("Data is empty after removing NaN values")
            return pd.Series([], index=pd.DatetimeIndex([]))

        original_data = data.copy()
        processed_data = data.copy()
        transformations_info = []

        try:
            for i, operation in enumerate(operations):
                op_type = operation.get('op', '').lower()

                if not op_type:
                    self.logger.warning(f"Operation {i + 1} has no 'op' field, skipping")
                    continue

                self.logger.info(f"Applying operation {i + 1}: {op_type}")

                if op_type == 'log':
                    # Перевірка на наявність нульових або від'ємних значень
                    if (processed_data <= 0).any():
                        min_positive = processed_data[processed_data > 0].min() if (processed_data > 0).any() else 1e-6
                        offset = abs(processed_data.min()) + min_positive if processed_data.min() <= 0 else 0
                        self.logger.warning(f"Negative or zero values found in data. Adding offset {offset}")
                        processed_data = processed_data + offset
                        transformations_info.append({
                            "method": "log",
                            "params": {"offset": offset}
                        })
                    else:
                        transformations_info.append({
                            "method": "log",
                            "params": {}
                        })
                    processed_data = np.log(processed_data)

                elif op_type == 'sqrt':
                    # Перевірка на наявність від'ємних значень
                    if (processed_data < 0).any():
                        offset = abs(processed_data.min()) + 1e-6
                        self.logger.warning(f"Negative values found in data. Adding offset {offset}")
                        processed_data = processed_data + offset
                        transformations_info.append({
                            "method": "sqrt",
                            "params": {"offset": offset}
                        })
                    else:
                        transformations_info.append({
                            "method": "sqrt",
                            "params": {}
                        })
                    processed_data = np.sqrt(processed_data)

                elif op_type == 'boxcox':
                    from scipy import stats
                    # BoxCox працює тільки з додатними значеннями
                    if (processed_data <= 0).any():
                        min_positive = processed_data[processed_data > 0].min() if (processed_data > 0).any() else 1e-6
                        offset = abs(processed_data.min()) + min_positive
                        self.logger.warning(f"Non-positive values found in data. Adding offset {offset}")
                        processed_data = processed_data + offset
                        processed_data, lambda_param = stats.boxcox(processed_data)
                        transformations_info.append({
                            "method": "boxcox",
                            "params": {
                                "lambda": lambda_param,
                                "offset": offset
                            }
                        })
                    else:
                        processed_data, lambda_param = stats.boxcox(processed_data)
                        transformations_info.append({
                            "method": "boxcox",
                            "params": {
                                "lambda": lambda_param
                            }
                        })

                elif op_type == 'diff':
                    order = operation.get('order', 1)
                    if not isinstance(order, int) or order < 1:
                        self.logger.warning(f"Invalid differencing order {order}, using 1 instead")
                        order = 1

                    processed_data = processed_data.diff(order).dropna()
                    transformations_info.append({
                        "method": "diff",
                        "params": {"order": order}
                    })

                elif op_type == 'seasonal_diff':
                    lag = operation.get('lag', 7)  # За замовчуванням тижнева сезонність
                    if not isinstance(lag, int) or lag < 1:
                        self.logger.warning(f"Invalid seasonal lag {lag}, using 7 instead")
                        lag = 7

                    processed_data = processed_data.diff(lag).dropna()
                    transformations_info.append({
                        "method": "seasonal_diff",
                        "params": {"lag": lag}
                    })

                elif op_type == 'remove_outliers':
                    method = operation.get('method', 'iqr')
                    threshold = operation.get('threshold', 1.5)

                    if method == 'iqr':
                        # Метод міжквартильного розмаху
                        q1 = processed_data.quantile(0.25)
                        q3 = processed_data.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr

                        # Зберігаємо інформацію про межі для можливого відновлення
                        outlier_mask = (processed_data < lower_bound) | (processed_data > upper_bound)
                        outlier_indices = outlier_mask[outlier_mask].index.tolist()
                        outlier_values = processed_data[outlier_mask].tolist()

                        # Заміна викидів на межі
                        processed_data = processed_data.clip(lower=lower_bound, upper=upper_bound)

                        transformations_info.append({
                            "method": "remove_outliers",
                            "params": {
                                "method": method,
                                "threshold": threshold,
                                "replaced_outliers": {
                                    "indices": outlier_indices,
                                    "values": outlier_values
                                }
                            }
                        })

                    elif method == 'zscore':
                        # Метод z-оцінки
                        z_scores = (processed_data - processed_data.mean()) / processed_data.std()
                        outlier_mask = abs(z_scores) > threshold

                        # Зберігаємо інформацію про викиди
                        outlier_indices = outlier_mask[outlier_mask].index.tolist()
                        outlier_values = processed_data[outlier_mask].tolist()

                        # Заміна викидів на None і інтерполяція
                        processed_data[outlier_mask] = None
                        processed_data = processed_data.interpolate(method='linear')

                        transformations_info.append({
                            "method": "remove_outliers",
                            "params": {
                                "method": method,
                                "threshold": threshold,
                                "replaced_outliers": {
                                    "indices": outlier_indices,
                                    "values": outlier_values
                                }
                            }
                        })

                    else:
                        self.logger.warning(f"Unknown outlier removal method: {method}, skipping")

                elif op_type == 'moving_average':
                    window = operation.get('window', 3)
                    center = operation.get('center', False)

                    if not isinstance(window, int) or window < 2:
                        self.logger.warning(f"Invalid window size {window} for moving average, using 3 instead")
                        window = 3

                    processed_data = processed_data.rolling(window=window, center=center).mean().dropna()
                    transformations_info.append({
                        "method": "moving_average",
                        "params": {
                            "window": window,
                            "center": center
                        }
                    })

                elif op_type == 'ewm':
                    # Експоненційно зважене середнє
                    span = operation.get('span', 5)

                    if not isinstance(span, int) or span < 2:
                        self.logger.warning(f"Invalid span {span} for EWM, using 5 instead")
                        span = 5

                    processed_data = processed_data.ewm(span=span).mean()
                    transformations_info.append({
                        "method": "ewm",
                        "params": {"span": span}
                    })

                elif op_type == 'normalize':
                    method = operation.get('method', 'minmax')

                    if method == 'minmax':
                        # Min-Max масштабування
                        min_val = processed_data.min()
                        max_val = processed_data.max()
                        if max_val > min_val:
                            processed_data = (processed_data - min_val) / (max_val - min_val)
                        else:
                            processed_data = processed_data * 0  # Якщо всі значення однакові

                        transformations_info.append({
                            "method": "normalize",
                            "params": {
                                "method": method,
                                "min": min_val,
                                "max": max_val
                            }
                        })

                    elif method == 'zscore':
                        # Z-score стандартизація
                        mean_val = processed_data.mean()
                        std_val = processed_data.std()
                        if std_val > 0:
                            processed_data = (processed_data - mean_val) / std_val
                        else:
                            processed_data = processed_data * 0  # Якщо стандартне відхилення нульове

                        transformations_info.append({
                            "method": "normalize",
                            "params": {
                                "method": method,
                                "mean": mean_val,
                                "std": std_val
                            }
                        })

                    else:
                        self.logger.warning(f"Unknown normalization method: {method}, skipping")

                else:
                    self.logger.warning(f"Unknown operation type: {op_type}, skipping")

                if len(processed_data) == 0:
                    self.logger.error(f"No data left after operation {i + 1}: {op_type}")
                    return pd.Series([], index=pd.DatetimeIndex([]))

            # Зберігаємо інформацію про трансформації, якщо є db_manager
            if self.db_manager is not None and transformations_info:
                try:
                    transformation_key = f"transform_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    self.db_manager.save_data_transformations(
                        key=transformation_key,
                        transformations=transformations_info,
                        metadata={
                            "original_length": len(original_data),
                            "processed_length": len(processed_data),
                            "timestamp": datetime.now().isoformat(),
                            "operations": operations
                        }
                    )
                    # Зберігаємо трансформації в локальному словнику також
                    self.transformations[transformation_key] = transformations_info

                    self.logger.info(f"Saved transformation pipeline with key: {transformation_key}")
                except Exception as db_error:
                    self.logger.error(f"Error saving transformation pipeline to database: {str(db_error)}")

            self.logger.info(
                f"Preprocessing pipeline applied successfully. Original length: {len(original_data)}, Processed length: {len(processed_data)}")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error during preprocessing pipeline: {str(e)}")
            return pd.Series([], index=pd.DatetimeIndex([]))

    def extract_volatility(self, data: pd.Series, window: int = 20) -> pd.Series:

        self.logger.info(f"Calculating volatility with window size {window}")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before volatility calculation.")
            data = data.dropna()

        if len(data) < window:
            self.logger.error(f"Not enough data points for volatility calculation with window={window}")
            return pd.Series([], index=pd.DatetimeIndex([]))

        try:
            # Розрахунок логарифмічних прибутків
            log_returns = np.log(data / data.shift(1)).dropna()

            # Розрахунок волатильності як ковзного стандартного відхилення логарифмічних прибутків
            volatility = log_returns.rolling(window=window).std()

            # Переведення стандартного відхилення у волатильність (анулізована)
            # Для різних частот даних множник буде різним:
            # - Денні дані: множник = sqrt(252) - кількість торгових днів у році
            # - Годинні дані: множник = sqrt(252 * 24)
            # - Хвилинні дані: множник = sqrt(252 * 24 * 60)

            # За замовчуванням припускаємо денні дані
            if isinstance(data.index, pd.DatetimeIndex):
                # Визначаємо частоту даних
                if len(data) >= 2:
                    time_diff = data.index[1:] - data.index[:-1]
                    median_diff = pd.Series(time_diff).median()

                    if median_diff <= pd.Timedelta(minutes=5):
                        annualization_factor = np.sqrt(252 * 24 * 12)  # 5-хвилинні дані
                    elif median_diff <= pd.Timedelta(hours=1):
                        annualization_factor = np.sqrt(252 * 24)  # Годинні дані
                    elif median_diff <= pd.Timedelta(days=1):
                        annualization_factor = np.sqrt(252)  # Денні дані
                    else:
                        annualization_factor = 1  # Не анулізуємо для нестандартних інтервалів
                else:
                    annualization_factor = np.sqrt(252)  # За замовчуванням
            else:
                annualization_factor = np.sqrt(252)  # За замовчуванням

            volatility = volatility * annualization_factor

            self.logger.info(f"Volatility calculation completed. Annualization factor: {annualization_factor}")

            return volatility

        except Exception as e:
            self.logger.error(f"Error during volatility calculation: {str(e)}")
            return pd.Series([], index=data.index)

    def run_auto_forecast(self, data: pd.Series, test_size: float = 0.2,
                          forecast_steps: int = 24, symbol: str = 'auto') -> Dict:

        self.logger.info(f"Starting auto forecasting process for symbol: {symbol}")

        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them before auto forecasting.")
            data = data.dropna()

        if len(data) < 30:  # Мінімальна кількість точок для змістовного аналізу
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
            # Генеруємо унікальний ключ для моделі
            model_key = f"{symbol}_auto_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # 1. Розділення даних на тренувальні та тестові
            train_size = int(len(data) * (1 - test_size))
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            self.logger.info(f"Split data: train={len(train_data)}, test={len(test_data)}")

            # 2. Перевірка стаціонарності та підготовка даних
            stationarity_check = self.check_stationarity(train_data)

            # Змінна для збереження інформації про трансформації
            transformations = []
            transformed_data = train_data.copy()

            # Якщо ряд нестаціонарний, застосовуємо перетворення
            if not stationarity_check["is_stationary"]:
                self.logger.info("Time series is non-stationary. Applying transformations.")

                # a) Логарифмічне перетворення, якщо всі дані > 0
                if all(train_data > 0):
                    self.logger.info("Applying log transformation")
                    transformed_data = np.log(transformed_data)
                    transformations.append({"op": "log"})

                    # Перевіряємо стаціонарність після логарифмування
                    log_stationary = self.check_stationarity(transformed_data)["is_stationary"]

                    if not log_stationary:
                        # б) Якщо все ще нестаціонарний, застосовуємо диференціювання
                        self.logger.info("Series still non-stationary. Applying differencing.")
                        transformed_data = self.difference_series(transformed_data, order=1)
                        transformations.append({"op": "diff", "order": 1})
                else:
                    # Якщо є від'ємні значення, відразу застосовуємо диференціювання
                    self.logger.info("Series contains non-positive values. Applying differencing directly.")
                    transformed_data = self.difference_series(train_data, order=1)
                    transformations.append({"op": "diff", "order": 1})

            # 3. Визначення наявності сезонності

            # Евристика для виявлення сезонності через автокореляцію
            from statsmodels.tsa.stattools import acf

            seasonal = False
            seasonal_period = None

            if len(transformed_data) > 50:  # Достатньо даних для аналізу сезонності
                max_lag = min(len(transformed_data) // 2, 365)  # Обмежуємо максимальний лаг
                acf_vals = acf(transformed_data, nlags=max_lag, fft=True)

                # Шукаємо піки в автокореляції (потенційні сезонні періоди)
                potential_periods = []

                # Перевіряємо типові періоди для фінансових даних
                for period in [7, 14, 30, 90, 180, 365]:
                    if period < len(acf_vals):
                        if acf_vals[period] > 0.3:  # Значна автокореляція
                            potential_periods.append((period, acf_vals[period]))

                if potential_periods:
                    # Вибираємо період з найсильнішою автокореляцією
                    potential_periods.sort(key=lambda x: x[1], reverse=True)
                    seasonal = True
                    seasonal_period = potential_periods[0][0]
                    self.logger.info(f"Detected seasonality with period: {seasonal_period}")

            # 4. Пошук оптимальних параметрів моделі
            if seasonal and seasonal_period:
                # Для сезонного ряду
                optimal_params = self.find_optimal_params(
                    transformed_data,
                    max_p=3, max_d=1, max_q=3,
                    seasonal=True
                )
            else:
                # Для несезонного ряду
                optimal_params = self.find_optimal_params(
                    transformed_data,
                    max_p=5, max_d=1, max_q=5,
                    seasonal=False
                )

            if optimal_params["status"] == "error":
                self.logger.error(f"Parameter search failed: {optimal_params['message']}")
                return {
                    "status": "error",
                    "message": f"Parameter search failed: {optimal_params['message']}",
                    "model_key": None,
                    "forecasts": None,
                    "performance": None
                }

            # 5. Навчання моделі з оптимальними параметрами
            model_info = None

            if seasonal and seasonal_period:
                # SARIMA model
                order = optimal_params["parameters"]["order"]
                seasonal_order = optimal_params["parameters"]["seasonal_order"]

                fit_result = self.fit_sarima(
                    transformed_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    symbol=symbol
                )

                model_type = "SARIMA"
            else:
                # ARIMA model
                order = optimal_params["parameters"]["order"]

                fit_result = self.fit_arima(
                    transformed_data,
                    order=order,
                    symbol=symbol
                )

                model_type = "ARIMA"

            if fit_result["status"] == "error":
                self.logger.error(f"Model fitting failed: {fit_result['message']}")
                return {
                    "status": "error",
                    "message": f"Model fitting failed: {fit_result['message']}",
                    "model_key": model_key,
                    "forecasts": None,
                    "performance": None
                }

            # Зберігаємо ключ навченої моделі
            model_key = fit_result["model_key"]
            model_info = fit_result["model_info"]

            # 6. Виконання прогнозу
            model_obj = self.models[model_key]["fit_result"]

            # Прогнозуємо на тестових даних (якщо є)
            if len(test_data) > 0:
                try:
                    # Прогноз тестового періоду для оцінки
                    test_forecast = model_obj.forecast(len(test_data))

                    # Для сезонних моделей або моделей з диференціюванням потрібно
                    # врахувати початкові значення при інверсному перетворенні
                    if "diff" in [t["op"] for t in transformations]:
                        # Цей блок потребує ретельної реалізації інверсних трансформацій
                        # Спрощений підхід - порівнюємо тренди
                        test_performance = {
                            "mse": mean_squared_error(test_data.values, test_forecast),
                            "rmse": np.sqrt(mean_squared_error(test_data.values, test_forecast)),
                            "mae": mean_absolute_error(test_data.values, test_forecast)
                        }

                        # Розрахунок MAPE, якщо немає нульових значень
                        if all(test_data != 0):
                            mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                            test_performance["mape"] = mape
                    else:
                        # Для моделей без диференціювання порівнюємо безпосередньо
                        test_performance = {
                            "mse": mean_squared_error(test_data.values, test_forecast),
                            "rmse": np.sqrt(mean_squared_error(test_data.values, test_forecast)),
                            "mae": mean_absolute_error(test_data.values, test_forecast)
                        }

                        # Розрахунок MAPE, якщо немає нульових значень
                        if all(test_data != 0):
                            mape = np.mean(np.abs((test_data.values - test_forecast) / test_data.values)) * 100
                            test_performance["mape"] = mape
                except Exception as e:
                    self.logger.error(f"Error during test forecast: {str(e)}")
                    test_performance = {"error": str(e)}
            else:
                test_performance = None

            # 7. Прогноз на майбутні періоди
            try:
                future_forecast = model_obj.forecast(forecast_steps)

                # Створюємо індекс для прогнозу
                if isinstance(data.index, pd.DatetimeIndex):
                    last_date = data.index[-1]

                    # Визначення інтервалу даних для створення правильного індексу прогнозу
                    if len(data) >= 2:
                        freq = pd.infer_freq(data.index)
                        if freq:
                            forecast_index = pd.date_range(start=last_date + pd.Timedelta(seconds=1),
                                                           periods=forecast_steps,
                                                           freq=freq)
                        else:
                            # Якщо не вдалося визначити частоту, оцінюємо середній інтервал
                            time_diff = data.index[1:] - data.index[:-1]
                            median_diff = pd.Series(time_diff).median()
                            forecast_index = pd.date_range(start=last_date + median_diff,
                                                           periods=forecast_steps,
                                                           freq=median_diff)
                    else:
                        # Якщо недостатньо точок, припускаємо денний інтервал
                        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                                       periods=forecast_steps)
                else:
                    # Для числових індексів
                    last_idx = data.index[-1]
                    if len(data) >= 2:
                        idx_diff = data.index[1] - data.index[0]
                    else:
                        idx_diff = 1
                    forecast_index = pd.RangeIndex(start=last_idx + idx_diff,
                                                   stop=last_idx + idx_diff * (forecast_steps + 1),
                                                   step=idx_diff)

                # Створюємо Series для прогнозу з правильним індексом
                future_forecast = pd.Series(future_forecast, index=forecast_index)

                # 8. Зворотні перетворення прогнозу (якщо застосовувались трансформації)
                for transform in reversed(transformations):
                    if transform["op"] == "diff":
                        # Для зворотного диференціювання потрібно початкове значення
                        # Беремо останнє значення вихідного ряду
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

            # 9. Збереження результатів у базу даних (якщо доступно)
            if self.db_manager is not None:
                try:
                    # Збираємо всі дані для збереження
                    forecast_data = {
                        "future_forecast": future_forecast.to_dict(),
                        "forecast_steps": forecast_steps,
                        "forecast_date": datetime.now().isoformat()
                    }

                    # Зберігаємо прогноз
                    self.db_manager.save_model_forecasts(model_key, forecast_data)

                    if test_performance:
                        # Зберігаємо метрики ефективності
                        self.db_manager.save_model_metrics(model_key, test_performance)

                    # Зберігаємо інформацію про трансформації даних
                    self.db_manager.save_data_transformations(model_key, {"transformations": transformations})

                    self.logger.info(f"Model {model_key} and forecast data saved to database")
                except Exception as db_error:
                    self.logger.error(f"Error saving to database: {str(db_error)}")

            # 10. Формуємо результат
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

    def load_crypto_data(self, db_manager: Any,
                         symbol: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         interval: str = '1d') -> pd.DataFrame:

        try:
            self.logger.info(f"Loading {symbol} data with interval {interval} from database")

            # Якщо кінцева дата не вказана, використовуємо поточну дату
            if end_date is None:
                end_date = datetime.now()
                self.logger.debug(f"End date not specified, using current date: {end_date}")

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Отримуємо дані з бази даних
            klines_data = manager.get_klines_processed(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )

            if klines_data is None or klines_data.empty:
                self.logger.warning(f"No data found for {symbol} with interval {interval}")
                return pd.DataFrame()

            # Перевіряємо наявність часового індексу
            if not isinstance(klines_data.index, pd.DatetimeIndex):
                # Шукаємо колонку з часовим індексом (timestamp, time, date, etc.)
                time_cols = [col for col in klines_data.columns if any(
                    time_str in col.lower() for time_str in ['time', 'date', 'timestamp'])]

                if time_cols:
                    # Використовуємо першу знайдену колонку часу
                    klines_data = klines_data.set_index(pd.DatetimeIndex(pd.to_datetime(klines_data[time_cols[0]])))
                    self.logger.info(f"Set index using column: {time_cols[0]}")
                else:
                    self.logger.warning("No time column found in data. Using default index.")

            # Сортуємо дані за часовим індексом
            klines_data = klines_data.sort_index()

            # Виводимо інформацію про отримані дані
            self.logger.info(f"Loaded {len(klines_data)} records for {symbol} "
                             f"from {klines_data.index.min()} to {klines_data.index.max()}")

            return klines_data

        except Exception as e:
            self.logger.error(f"Error loading crypto data: {str(e)}")
            raise

    def save_forecast_to_db(self, db_manager: Any, symbol: str,
                            forecast_data: pd.Series, model_key: str) -> bool:

        try:
            self.logger.info(f"Saving forecast for {symbol} using model {model_key}")

            if forecast_data is None or len(forecast_data) == 0:
                self.logger.error("No forecast data provided")
                return False

            # Перевіряємо чи forecast_data є pd.Series
            if not isinstance(forecast_data, pd.Series):
                try:
                    forecast_data = pd.Series(forecast_data)
                    self.logger.warning("Converted forecast data to pandas Series")
                except Exception as convert_error:
                    self.logger.error(f"Could not convert forecast data to pandas Series: {str(convert_error)}")
                    return False

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return False

            # Перевіряємо наявність моделі в базі даних
            model_info = manager.get_model_by_key(model_key)

            if model_info is None:
                self.logger.error(f"Model with key {model_key} not found in database")
                return False

            # Перетворюємо дані прогнозу у формат для збереження
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

            # Зберігаємо прогноз у базі даних
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

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return None

            # Перевіряємо наявність моделі в базі даних
            model_info = manager.get_model_by_key(model_key)

            if model_info is None:
                self.logger.warning(f"Model with key {model_key} not found in database")
                return None

            # Отримуємо прогнози з бази даних
            forecast_dict = manager.get_model_forecasts(model_key)

            if not forecast_dict:
                self.logger.warning(f"No forecasts found for model {model_key}")
                return None

            # Перевіряємо, чи є прогнози для заданого символу
            if symbol.upper() != forecast_dict.get('symbol', '').upper():
                self.logger.warning(f"Forecast for symbol {symbol} not found in model {model_key}")
                return None

            # Перетворюємо словник прогнозів на pd.Series
            try:
                forecast_data = forecast_dict.get('forecast_data', {})

                # Перетворюємо ключі на datetime, якщо вони є датами
                index = []
                values = []

                for timestamp_str, value in forecast_data.items():
                    try:
                        # Спробуємо перетворити на datetime
                        timestamp = pd.to_datetime(timestamp_str)
                    except:
                        # Якщо не вийшло, використовуємо як є
                        timestamp = timestamp_str

                    index.append(timestamp)
                    values.append(float(value))

                # Створюємо pandas Series з правильним індексом
                forecast_series = pd.Series(values, index=index)

                # Сортуємо за індексом
                forecast_series = forecast_series.sort_index()

                self.logger.info(f"Successfully loaded forecast with {len(forecast_series)} points for {symbol}")

                return forecast_series

            except Exception as e:
                self.logger.error(f"Error converting forecast data to Series: {str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading forecast from database: {str(e)}")
            return None

    def get_available_crypto_symbols(self, db_manager: Any) -> List[str]:

        try:
            self.logger.info("Getting available cryptocurrency symbols from database")

            # Використовуємо переданий db_manager або збережений в класі
            manager = db_manager if db_manager is not None else self.db_manager

            if manager is None:
                error_msg = "Database manager not available. Please provide a valid db_manager."
                self.logger.error(error_msg)
                return []

            # Отримуємо список доступних символів
            symbols = manager.get_available_symbols()

            if symbols is None:
                self.logger.warning("No symbols returned from database")
                return []

            # Перевіряємо, що отримані дані є списком
            if not isinstance(symbols, list):
                try:
                    symbols = list(symbols)
                except Exception as e:
                    self.logger.error(f"Could not convert symbols to list: {str(e)}")
                    return []

            # Перевіряємо, що символи не порожні
            symbols = [s for s in symbols if s]

            # Видаляємо дублікати і сортуємо
            symbols = sorted(set(symbols))

            self.logger.info(f"Found {len(symbols)} available cryptocurrency symbols")

            return symbols

        except Exception as e:
            self.logger.error(f"Error getting available cryptocurrency symbols: {str(e)}")
            return []

    def get_last_update_time(self, db_manager: Any, symbol: str,
                             interval: str = '1d') -> Optional[datetime]:

        self.logger.info(f"Getting last update time for {symbol} with interval {interval}")

        if db_manager is None:
            self.logger.error("Database manager is not provided")
            return None

        try:
            # Перевіряємо, чи переданий db_manager був заданий при ініціалізації класу
            # якщо ні, використовуємо переданий
            db = self.db_manager if self.db_manager is not None else db_manager

            # Припускаємо, що в db_manager є метод для отримання останніх даних свічок
            latest_kline = db.get_latest_kline(symbol=symbol, interval=interval)

            if latest_kline is not None and hasattr(latest_kline, 'timestamp'):
                # Якщо отримали дані і є відмітка часу
                self.logger.info(f"Last update time for {symbol} ({interval}): {latest_kline.timestamp}")
                return latest_kline.timestamp

            # Якщо немає прямого методу, спробуємо отримати через оброблені свічки
            klines_data = db.get_klines_processed(
                symbol=symbol,
                interval=interval,
                limit=1,  # Беремо тільки одну (останню) свічку
                sort_order="DESC"  # Сортуємо за спаданням дати
            )

            if klines_data is not None and not klines_data.empty:
                # Отримуємо індекс останньої свічки (який повинен бути datetime)
                if isinstance(klines_data.index[0], datetime):
                    last_update = klines_data.index[0]
                else:
                    # Якщо індекс не datetime, спробуємо знайти стовпець з часовою міткою
                    for col in ['timestamp', 'time', 'date', 'datetime']:
                        if col in klines_data.columns:
                            last_update = klines_data[col].iloc[0]
                            if not isinstance(last_update, datetime):
                                # Конвертуємо в datetime, якщо це не datetime
                                if isinstance(last_update, (int, float)):
                                    # Припускаємо, що це UNIX timestamp в мілісекундах або секундах
                                    if last_update > 1e11:  # Якщо в мілісекундах
                                        last_update = datetime.fromtimestamp(last_update / 1000)
                                    else:  # Якщо в секундах
                                        last_update = datetime.fromtimestamp(last_update)
                                else:
                                    # Якщо це строка, пробуємо парсити
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

        # Ініціалізуємо словник для результатів
        results = {}

        # Встановлюємо значення за замовчуванням для дат
        if end_date is None:
            end_date = datetime.now()
            self.logger.info(f"End date not provided, using current time: {end_date}")

        # Обробляємо кожен символ
        for symbol in symbols:
            self.logger.info(f"Processing symbol: {symbol}")

            try:
                # Перевіряємо, чи є дані для цього символу
                if not self._check_symbol_data_available(db_manager, symbol, interval):
                    self.logger.warning(f"No data available for {symbol}, skipping")
                    results[symbol] = {
                        "status": "error",
                        "message": f"No data available for {symbol}",
                        "timestamp": datetime.now()
                    }
                    continue

                # Якщо початкова дата не вказана, отримуємо останню дату оновлення
                # і віднімаємо певний період (наприклад, 365 днів для денних даних)
                if start_date is None:
                    last_update = self.get_last_update_time(db_manager, symbol, interval)
                    if last_update is not None:
                        if interval == '1d':
                            start_date = last_update - timedelta(days=365)  # Рік даних для денного інтервалу
                        elif interval == '1h':
                            start_date = last_update - timedelta(days=30)  # 30 днів для годинного інтервалу
                        elif interval in ['15m', '5m', '1m']:
                            start_date = last_update - timedelta(days=7)  # Тиждень для хвилинних інтервалів
                        else:
                            start_date = last_update - timedelta(days=180)  # Півроку за замовчуванням

                        self.logger.info(f"Calculated start date for {symbol}: {start_date}")
                    else:
                        self.logger.warning(f"Cannot determine last update time for {symbol}, using default")
                        # За замовчуванням, беремо дані за останній рік
                        start_date = end_date - timedelta(days=365)

                # Завантажуємо дані для аналізу
                data = self.load_crypto_data(
                    db_manager=db_manager,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )

                if data is None or data.empty:
                    self.logger.warning(f"No data loaded for {symbol}, skipping")
                    results[symbol] = {
                        "status": "error",
                        "message": f"No data loaded for {symbol}",
                        "timestamp": datetime.now()
                    }
                    continue

                # Вибираємо цільову колонку для аналізу (зазвичай 'close')
                target_column = 'close'
                if target_column not in data.columns:
                    # Шукаємо альтернативу, якщо 'close' немає
                    possible_columns = ['Close', 'price', 'Price', 'value', 'Value']
                    for col in possible_columns:
                        if col in data.columns:
                            target_column = col
                            break
                    else:
                        # Якщо немає відповідної колонки, використовуємо першу колонку з числовими даними
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

                # Запускаємо автоматичне прогнозування
                forecast_result = self.run_auto_forecast(
                    data=data[target_column],
                    test_size=0.2,  # 20% даних для тестування
                    forecast_steps=24,  # Прогноз на 24 періоди вперед
                    symbol=symbol
                )

                # Зберігаємо результати прогнозування
                if forecast_result.get("status") == "success" and "model_key" in forecast_result:
                    model_key = forecast_result["model_key"]

                    # Зберігаємо комплексну інформацію про модель в БД
                    if self.db_manager is not None:
                        try:
                            self.db_manager.save_complete_model(model_key, forecast_result.get("model_info", {}))
                            self.logger.info(f"Model for {symbol} saved to database with key {model_key}")
                        except Exception as db_error:
                            self.logger.error(f"Error saving model for {symbol} to database: {str(db_error)}")

                    # Додаємо результат до загального словника
                    results[symbol] = {
                        "status": "success",
                        "message": f"Successfully processed {symbol}",
                        "model_key": model_key,
                        "timestamp": datetime.now(),
                        **forecast_result
                    }
                else:
                    # Якщо прогнозування не вдалося, додаємо інформацію про помилку
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

        # Додаємо загальну статистику
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
            # Перевіряємо, чи є такий символ у списку доступних
            available_symbols = self.get_available_crypto_symbols(db_manager)
            if symbol not in available_symbols:
                self.logger.warning(f"Symbol {symbol} not in available symbols list")
                return False

            # Перевіряємо, чи є хоча б деякі дані для цього символу
            last_update = self.get_last_update_time(db_manager, symbol, interval)
            if last_update is None:
                self.logger.warning(f"No last update time for {symbol}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking data availability for {symbol}: {str(e)}")
            return False
def main():
    import pprint

    symbol = "BTCUSDT"
    interval = "1d"
    forecast_steps = 7

    model = TimeSeriesModels()
    db = model.db_manager

    if db is None:
        print(" Менеджер бази даних не налаштований.")
        return

    try:
        df = db.get_klines_processed(symbol, interval)
    except Exception as e:
        print(f" Помилка при отриманні даних: {e}")
        return

    if df is None or df.empty or "close" not in df.columns:
        print(" Немає даних для тренування моделі.")
        return

    price_series = df["close"]

    stat_info = model.check_stationarity(price_series)
    if not stat_info["is_stationary"]:
        price_series = model.difference_series(price_series)

    #  ARIMA
    arima_key = None
    arima_params = model.find_optimal_params(price_series, seasonal=False)
    if arima_params["status"] == "success":
        arima_result = model.fit_arima(price_series, arima_params["parameters"]["order"], symbol=symbol)
        arima_key = arima_result["model_key"]
        print(f" ARIMA збережено як {arima_key}")
    else:
        print("️ ARIMA: не вдалося підібрати параметри")

    #  SARIMA
    sarima_key = None
    sarima_params = model.find_optimal_params(price_series, seasonal=True)
    if sarima_params["status"] == "success":
        sarima_result = model.fit_sarima(
            price_series,
            order=sarima_params["parameters"]["order"],
            seasonal_order=sarima_params["parameters"]["seasonal_order"],
            symbol=symbol
        )
        sarima_key = sarima_result["model_key"]
        print(f" SARIMA збережено як {sarima_key}")
    else:
        print(" SARIMA: не вдалося підібрати параметри")

    #  Порівняння моделей
    if arima_key and sarima_key:
        aic_arima = arima_result["model_info"]["stats"]["aic"]
        aic_sarima = sarima_result["model_info"]["stats"]["aic"]
        better_key = arima_key if aic_arima < aic_sarima else sarima_key
        print(f"🏆 Краща модель за AIC: {better_key}")
    else:
        better_key = arima_key or sarima_key

    #  Прогнозування
    if better_key:
        forecast = model.forecast(better_key, steps=forecast_steps)
        if not forecast.empty:
            print("\n Прогноз:")
            pprint.pprint(forecast.to_dict())
        else:
            print(" Прогнозування не вдалося.")
    else:
        print(" Немає доступної моделі для прогнозування.")

    print("\n Завершено.")

if __name__ == "__main__":
    main()
