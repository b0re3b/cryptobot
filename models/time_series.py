import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta


class TimeSeriesModels:
    """
    Клас для моделювання часових рядів криптовалют з використанням класичних методів.

    Клас працює з базою даних через об'єкт db_manager, який має містити наступні методи роботи з БД:
    - get_klines_processed: для отримання оброблених свічок
    - get_volume_profile: для отримання профілю об'єму
    - save_model_metadata: для збереження метаданих моделі
    - save_model_parameters: для збереження параметрів моделі
    - save_model_metrics: для збереження метрик ефективності моделі
    - save_model_forecasts: для збереження прогнозів
    - save_model_binary: для збереження серіалізованої моделі
    - save_data_transformations: для збереження інформації про перетворення даних
    - save_complete_model: для комплексного збереження всіх даних моделі
    - get_model_by_key: для отримання інформації про модель за ключем
    - get_model_parameters: для отримання параметрів моделі
    - get_model_metrics: для отримання метрик ефективності моделі
    - get_model_forecasts: для отримання прогнозів моделі
    - load_model_binary: для завантаження серіалізованої моделі
    - get_data_transformations: для отримання інформації про перетворення даних
    - load_complete_model: для комплексного завантаження всіх даних моделі
    - get_models_by_symbol: для отримання всіх моделей для певного символу
    - get_latest_model_by_symbol: для отримання останньої моделі для певного символу
    - get_model_performance_history: для отримання історії продуктивності моделі
    - update_model_status: для оновлення статусу активності моделі
    - compare_model_forecasts: для порівняння прогнозів декількох моделей
    - get_model_forecast_accuracy: для розрахунку точності прогнозу
    - get_available_symbols: для отримання списку доступних символів криптовалют
    """

    def __init__(self, db_manager=None, log_level=logging.INFO):

        self.db_manager = db_manager
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

    def check_stationarity(self, data: pd.Series) -> Dict:

        from statsmodels.tsa.stattools import adfuller, kpss

        # Перевіряємо, що дані не містять NaN значень
        if data.isnull().any():
            self.logger.warning("Data contains NaN values. Removing them for stationarity check.")
            data = data.dropna()

        # Перевірка об'єму даних
        if len(data) < 20:
            self.logger.warning("Data series too short for reliable stationarity tests")
            return {
                "adf_test": {"is_stationary": None, "p_value": None, "test_statistic": None, "critical_values": None},
                "kpss_test": {"is_stationary": None, "p_value": None, "test_statistic": None, "critical_values": None},
                "rolling_statistics": {"mean_stationary": None, "std_stationary": None},
                "is_stationary": False
            }

        # Розрахунок рухомого середнього та стандартного відхилення
        rolling_mean = data.rolling(window=12).mean()
        rolling_std = data.rolling(window=12).std()

        # Розрахунок відносної зміни для рухомих статистик
        if len(data) > 24:
            mean_change_rel = abs(
                (rolling_mean.iloc[-12:].mean() - rolling_mean.iloc[12:24].mean()) / rolling_mean.iloc[12:24].mean())
            std_change_rel = abs(
                (rolling_std.iloc[-12:].mean() - rolling_std.iloc[12:24].mean()) / rolling_std.iloc[12:24].mean())
            mean_stationary = mean_change_rel < 0.1  # Вважаємо стаціонарним, якщо зміна < 10%
            std_stationary = std_change_rel < 0.1
        else:
            mean_stationary = None
            std_stationary = None

        # Тест Дікі-Фуллера (ADF-test) для перевірки наявності одиничного кореня
        try:
            adf_result = adfuller(data, autolag='AIC')
            adf_is_stationary = adf_result[1] < 0.05  # p-значення < 0.05 => стаціонарний ряд
        except Exception as e:
            self.logger.error(f"Error during ADF test: {e}")
            adf_result = [None, None, None, {}]
            adf_is_stationary = False

        # KPSS тест на тренд-стаціонарність
        try:
            kpss_result = kpss(data, regression='ct', nlags='auto')
            kpss_is_stationary = kpss_result[1] > 0.05  # p-значення > 0.05 => стаціонарний ряд
        except Exception as e:
            self.logger.error(f"Error during KPSS test: {e}")
            kpss_result = [None, None, None, {}]
            kpss_is_stationary = False

        # Загальний висновок про стаціонарність
        # Вважаємо ряд стаціонарним, якщо обидва тести підтверджують це
        is_stationary = adf_is_stationary and kpss_is_stationary

        result = {
            "adf_test": {
                "is_stationary": adf_is_stationary,
                "p_value": adf_result[1],
                "test_statistic": adf_result[0],
                "critical_values": adf_result[4]
            },
            "kpss_test": {
                "is_stationary": kpss_is_stationary,
                "p_value": kpss_result[1],
                "test_statistic": kpss_result[0],
                "critical_values": kpss_result[3]
            },
            "rolling_statistics": {
                "mean_stationary": mean_stationary,
                "std_stationary": std_stationary
            },
            "is_stationary": is_stationary
        }

        self.logger.info(f"Stationarity check completed: {is_stationary}")

        return result

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
            # 1. Перевірка стаціонарності ряду
            stationarity_result = self.check_stationarity(data)
            result["details"]["stationarity"] = stationarity_result

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
        """
        Аналіз залишків моделі.

        Args:
            model_key: Ключ моделі
            data: Дані для порівняння (якщо None, використовуються дані з моделі)

        Returns:
            Результати аналізу залишків

        Примітка:
        - Спочатку потрібно завантажити модель з БД за допомогою self.db_manager.load_complete_model(model_key)
        """
        pass

    def forecast_with_intervals(self, model_key: str, steps: int = 24,
                                alpha: float = 0.05) -> Dict:
        """
        Прогнозування з довірчими інтервалами.

        Args:
            model_key: Ключ моделі
            steps: Кількість кроків для прогнозу
            alpha: Рівень значущості для довірчих інтервалів

        Returns:
            Прогноз та довірчі інтервали

        Примітка:
        - Спочатку потрібно завантажити модель з БД за допомогою self.db_manager.load_complete_model(model_key)
        - Після прогнозування рекомендується зберегти результати в БД за допомогою
          self.db_manager.save_model_forecasts включно з нижніми та верхніми межами довірчих інтервалів
        """
        pass

    def compare_models(self, model_keys: List[str], test_data: pd.Series) -> Dict:
        """
        Порівняння декількох моделей за метриками точності.

        Args:
            model_keys: Список ключів моделей для порівняння
            test_data: Тестові дані для оцінки

        Returns:
            Словник з результатами порівняння моделей

        Примітка:
        - Для кожного ключа моделі потрібно завантажити модель з БД за допомогою self.db_manager.load_complete_model
        - Можна використати self.db_manager.compare_model_forecasts для порівняння прогнозів різних моделей
        """
        pass

    def apply_preprocessing_pipeline(self, data: pd.Series, operations: List[Dict]) -> pd.Series:
        """
        Застосування послідовності операцій препроцесингу до часового ряду.

        Args:
            data: Вхідний часовий ряд
            operations: Список словників з операціями та їх параметрами
                        Приклад: [{'op': 'log'}, {'op': 'diff', 'order': 1}]

        Returns:
            Оброблений часовий ряд

        Примітка:
        - Інформація про pipeline може бути збережена в БД за допомогою self.db_manager.save_data_transformations
          для подальшого використання при інверсній трансформації прогнозів
        """
        pass

    def extract_volatility(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        Розрахунок волатильності часового ряду.

        Args:
            data: Часовий ряд
            window: Розмір вікна для розрахунку волатильності

        Returns:
            Часовий ряд волатильності
        """
        pass

    def run_auto_forecast(self, data: pd.Series, test_size: float = 0.2,
                          forecast_steps: int = 24, symbol: str = 'auto') -> Dict:
        """
        Автоматичний пошук параметрів, навчання та прогнозування.

        Args:
            data: Часовий ряд
            test_size: Частка даних для тестування
            forecast_steps: Кількість кроків для прогнозу
            symbol: Ідентифікатор моделі

        Returns:
            Словник з результатами автоматичного прогнозування

        Примітка:
        - Результати прогнозування та сама модель можуть бути збережені в БД за допомогою
          self.db_manager.save_complete_model, який включає в себе збереження всіх компонентів моделі
        """
        pass

    def load_crypto_data(self, db_manager: Any,
                         symbol: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         interval: str = '1d') -> pd.DataFrame:
        """
        Завантаження даних криптовалюти з бази даних через DatabaseManager.

        Args:
            db_manager: Об'єкт класу DatabaseManager
            symbol: Символ криптовалюти (наприклад, 'BTC', 'ETH')
            start_date: Початкова дата (якщо None, береться найраніша в базі)
            end_date: Кінцева дата (якщо None, береться поточна дата)
            interval: Інтервал даних ('1m', '5m', '15m', '1h', '4h', '1d')

        Returns:
            DataFrame з даними криптовалюти

        Примітка:
        - Використовує self.db_manager.get_klines_processed для отримання оброблених свічок з БД
        """
        pass

    def save_forecast_to_db(self, db_manager: Any, symbol: str,
                            forecast_data: pd.Series, model_key: str) -> bool:
        """
        Збереження результатів прогнозу в базу даних.

        Args:
            db_manager: Об'єкт класу DatabaseManager
            symbol: Символ криптовалюти
            forecast_data: Дані прогнозу
            model_key: Ключ моделі, яка була використана

        Returns:
            Успішність операції

        Примітка:
        - Спочатку потрібно отримати ID моделі за допомогою self.db_manager.get_model_by_key(model_key)
        - Потім використати self.db_manager.save_model_forecasts для збереження прогнозів
        """
        pass

    def load_forecast_from_db(self, db_manager: Any, symbol: str,
                              model_key: str) -> Optional[pd.Series]:
        """
        Завантаження збереженого прогнозу з бази даних.

        Args:
            db_manager: Об'єкт класу DatabaseManager
            symbol: Символ криптовалюти
            model_key: Ключ моделі, яка була використана

        Returns:
            Дані прогнозу або None, якщо прогноз не знайдено

        Примітка:
        - Спочатку потрібно отримати ID моделі за допомогою self.db_manager.get_model_by_key(model_key)
        - Потім використати self.db_manager.get_model_forecasts для отримання прогнозів
        """
        pass

    def get_available_crypto_symbols(self, db_manager: Any) -> List[str]:
        """
        Отримання списку доступних символів криптовалют з бази даних.

        Args:
            db_manager: Об'єкт класу DatabaseManager

        Returns:
            Список доступних символів криптовалют

        Примітка:
        - Використовує self.db_manager.get_available_symbols для отримання списку символів криптовалют
        """
        pass

    def get_last_update_time(self, db_manager: Any, symbol: str,
                             interval: str = '1d') -> Optional[datetime]:
        """
        Отримання часу останнього оновлення даних криптовалюти.

        Args:
            db_manager: Об'єкт класу DatabaseManager
            symbol: Символ криптовалюти
            interval: Інтервал даних

        Returns:
            Дата і час останнього оновлення або None, якщо дані відсутні

        Примітка:
        - Може використовувати допоміжні методи db_manager для отримання цієї інформації
        """
        pass

    def batch_process_symbols(self, db_manager: Any, symbols: List[str],
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              interval: str = '1d') -> Dict[str, Dict]:
        """
        Пакетна обробка декількох криптовалют.

        Args:
            db_manager: Об'єкт класу DatabaseManager
            symbols: Список символів криптовалют
            start_date: Початкова дата
            end_date: Кінцева дата
            interval: Інтервал даних

        Returns:
            Словник з результатами для кожного символу

        Примітка:
        - Для кожного символу завантажує дані, навчає модель та зберігає результати в БД
        - Використовує self.load_crypto_data і self.run_auto_forecast для кожного символу
        - Зберігає всі результати в БД за допомогою self.db_manager.save_complete_model
        """
        pass