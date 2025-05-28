from datetime import datetime
from typing import Dict, Tuple, List
import decimal

import numpy as np
import pandas as pd
from pmdarima import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data.db import DatabaseManager
from timeseriesmodels.TimeSeriesAnalyzer import TimeSeriesAnalyzer
from utils.logger import CryptoLogger


class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.logger = CryptoLogger('ModelEvaluator')
        self.db_manager = DatabaseManager()
        self.analyzer = TimeSeriesAnalyzer()

    def evaluate_model(self, model_key: str, test_data: pd.Series, use_rolling_validation: bool = True,
                       window_size: int = 100, step: int = 20, forecast_horizon: int = 10,
                       apply_inverse_transforms: bool = False) -> Dict:
        """
        Оцінює модель, порівнюючи її прогнози з тестовими даними.
        Має можливість додатково оцінити модель за допомогою методу ковзного вікна для більш надійної валідації.

        Args:
            model_key: Ключ моделі
            test_data: Часовий ряд з тестовими даними
            use_rolling_validation: Чи використовувати метод ковзного вікна для валідації
            window_size: Розмір вікна для валідації методом ковзного вікна
            step: Крок зсуву для валідації методом ковзного вікна
            forecast_horizon: Горизонт прогнозування для валідації методом ковзного вікна
            apply_inverse_transforms: Чи застосовувати зворотні трансформації до прогнозів

        Returns:
            Dictionary з метриками оцінки моделі
        """
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

            # Визначаємо тип моделі
            model_type = model_info.get("metadata", {}).get("model_type", "ARIMA")
            model_params = model_info.get("parameters", {})

            # Базова оцінка на тестових даних
            base_evaluation_result = self._evaluate_on_test_data(
                model_key, fit_result, test_data, model_type, apply_inverse_transforms
            )

            result = {
                "status": "success",
                "message": "Model evaluation completed successfully",
                "metrics": base_evaluation_result["metrics"],
                "visual_data": base_evaluation_result["visual_data"]
            }

            # Додаткова оцінка за допомогою методу ковзного вікна, якщо вказано
            if use_rolling_validation:
                self.logger.info(f"Performing additional rolling window validation for model {model_key}")

                # Отримуємо параметри моделі для ковзної валідації
                if model_type.upper() == "ARIMA":
                    order = model_params.get("order", (1, 1, 1))
                    seasonal_order = None
                    rolling_model_type = "arima"
                elif model_type.upper() == "SARIMA":
                    order = model_params.get("order", (1, 1, 1))
                    seasonal_order = model_params.get("seasonal_order", (1, 0, 1, 7))
                    rolling_model_type = "sarima"
                else:
                    self.logger.warning(f"Unknown model type for rolling validation: {model_type}, using ARIMA")
                    order = (1, 1, 1)
                    seasonal_order = None
                    rolling_model_type = "arima"

                # Виконуємо валідацію методом ковзного вікна
                rolling_validation_result = self.rolling_window_validation(
                    data=test_data,
                    model_type=rolling_model_type,
                    order=order,
                    seasonal_order=seasonal_order,
                    window_size=window_size,
                    step=step,
                    forecast_horizon=forecast_horizon
                )

                if rolling_validation_result["status"] == "success" or rolling_validation_result["status"] == "warning":
                    # Додаємо результати ковзної валідації до загального результату
                    result["rolling_validation"] = {
                        "status": rolling_validation_result["status"],
                        "message": rolling_validation_result["message"],
                        "iterations": rolling_validation_result["iterations"],
                        "aggregated_metrics": rolling_validation_result.get("aggregated_metrics", {})
                    }

                    # Зберігаємо інформацію про найкращу та найгіршу ітерації
                    if "metrics" in rolling_validation_result and "rmse" in rolling_validation_result["metrics"]:
                        rmse_values = rolling_validation_result["metrics"]["rmse"]
                        if rmse_values:
                            best_iter_idx = np.argmin(rmse_values)
                            worst_iter_idx = np.argmax(rmse_values)

                            result["rolling_validation"]["best_iteration"] = {
                                "index": int(best_iter_idx),
                                "rmse": float(rmse_values[best_iter_idx])
                            }

                            result["rolling_validation"]["worst_iteration"] = {
                                "index": int(worst_iter_idx),
                                "rmse": float(rmse_values[worst_iter_idx])
                            }

                    # Збагачуємо основні метрики інформацією про стабільність моделі
                    if "aggregated_metrics" in rolling_validation_result:
                        agg_metrics = rolling_validation_result["aggregated_metrics"]
                        result["metrics"].update({
                            "stability": {
                                "rmse_std": agg_metrics.get("std_rmse"),
                                "rmse_range": agg_metrics.get("max_rmse") - agg_metrics.get("min_rmse"),
                                "coefficient_of_variation": agg_metrics.get("std_rmse") / agg_metrics.get("mean_rmse")
                                if agg_metrics.get("mean_rmse") else None
                            }
                        })

                        # Додаємо порівняння з базовою оцінкою
                        if "rmse" in result["metrics"]:
                            base_rmse = result["metrics"]["rmse"]
                            rolling_mean_rmse = agg_metrics.get("mean_rmse")

                            if rolling_mean_rmse:
                                rmse_ratio = base_rmse / rolling_mean_rmse
                                result["metrics"]["stability"]["base_to_rolling_rmse_ratio"] = rmse_ratio

                                # Оцінка надійності моделі
                                if rmse_ratio < 0.8:
                                    reliability_assessment = "Основна оцінка оптимістична порівняно з ковзною валідацією"
                                elif rmse_ratio > 1.2:
                                    reliability_assessment = "Основна оцінка песимістична порівняно з ковзною валідацією"
                                else:
                                    reliability_assessment = "Оцінки узгоджені, модель стабільна"

                                result["metrics"]["stability"]["reliability_assessment"] = reliability_assessment
                else:
                    self.logger.warning(f"Rolling validation failed: {rolling_validation_result['message']}")
                    result["rolling_validation"] = {
                        "status": "error",
                        "message": rolling_validation_result["message"]
                    }

            # Зберігаємо метрики в БД, якщо є підключення
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Saving metrics for model {model_key} to database")
                    self.db_manager.save_model_metrics(model_key, result["metrics"])
                    self.logger.info(f"Metrics for model {model_key} saved successfully")
                except Exception as e:
                    self.logger.error(f"Error saving metrics to database: {str(e)}")

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

    def _evaluate_on_test_data(self, model_key: str, fit_result, test_data: pd.Series,
                              model_type: str, apply_inverse_transforms: bool = False) -> Dict:
        """
        Допоміжний метод для оцінки моделі на тестових даних.

        Args:
            model_key: Ключ моделі
            fit_result: Результат навчання моделі
            test_data: Тестові дані
            model_type: Тип моделі (ARIMA, SARIMA, тощо)
            apply_inverse_transforms: Чи застосовувати зворотні трансформації

        Returns:
            Dictionary з результатами оцінки
        """
        # Отримуємо прогноз для тестового періоду
        steps = len(test_data)
        self.logger.info(f"Generating in-sample forecasts for {steps} test points")

        # Генеруємо прогноз в залежності від типу моделі
        if model_type.upper() == "ARIMA":
            # Для ARIMA використовуємо прямий метод forecast
            forecast = fit_result.forecast(steps=steps)
        elif model_type.upper() == "SARIMA":
            # Для SARIMA використовуємо get_forecast
            forecast_result = fit_result.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
        else:
            self.logger.warning(f"Unknown model type: {model_type}, trying generic forecast method")
            try:
                forecast = fit_result.forecast(steps=steps)
            except Exception as e:
                self.logger.error(f"Error using generic forecast: {str(e)}")
                raise ValueError(f"Unable to generate forecast for model type: {model_type}")

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

        # Застосування зворотних трансформацій, тільки якщо явно вказано
        if apply_inverse_transforms and self.db_manager is not None:
            self.logger.info("Applying inverse transformations to forecast")
            transformations = self.db_manager.get_data_transformations(model_key)
            if transformations:
                for transform in reversed(transformations):
                    if transform.get("method"):
                        self.logger.info(f"Applying inverse transformation: {transform['method']}")
                        try:
                            if transform["method"] == "log":
                                forecast_series = np.exp(forecast_series)
                            elif transform["method"] == "boxcox":
                                from scipy import special
                                forecast_series = special.inv_boxcox(forecast_series, transform.get("lambda", 0))
                            elif transform["method"] == "sqrt":
                                forecast_series = forecast_series ** 2
                        except Exception as e:
                            self.logger.error(f"Error applying inverse transformation {transform['method']}: {str(e)}")
                            break
            else:
                self.logger.info("No transformations found for inverse application")

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
        # Покращена обробка нульових значень для MAPE
        try:
            # Створюємо маску для ненульових значень
            mask = np.abs(y_true) > 1e-10  # Значення більше маленького порогу

            if np.any(mask):
                # Розрахунок MAPE лише на ненульових значеннях
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
                self.logger.info(f"MAPE calculated on {np.sum(mask)}/{len(mask)} non-zero values")
            else:
                self.logger.warning("All true values are close to zero, using alternative metric to MAPE")
                # Альтернативний показник - використовуємо MAE, нормалізований до середнього значення прогнозу
                # або деякого малого значення
                denominator = max(np.mean(np.abs(y_pred)), 1e-10)
                mape = (mae / denominator) * 100
                self.logger.info(f"Using alternative to MAPE: (MAE / mean_prediction) * 100 = {mape:.2f}%")
        except Exception as e:
            self.logger.warning(f"Error calculating MAPE: {str(e)}. Using alternative method.")
            # Альтернативний підхід для запобігання ділення на нуль
            epsilon = max(np.mean(np.abs(y_true)), np.finfo(float).eps)  # Використовуємо середнє або мале число
            mape = (mae / epsilon) * 100

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
            "nonzero_sample_size": int(np.sum(mask)) if 'mask' in locals() else None,
            "evaluation_date": datetime.now().isoformat()
        }

        return {
            "metrics": metrics,
            "visual_data": {
                "actuals": test_data.tolist(),
                "predictions": forecast_series.tolist(),
                "dates": [str(idx) for idx in test_data.index]
            }
        }
    def rolling_window_validation(self, data: pd.Series, model_type: str = 'arima',
                                  order: Tuple = None, seasonal_order: Tuple = None,
                                  window_size: int = 100, step: int = 20,
                                  forecast_horizon: int = 10) -> Dict:
        """
        Проводить валідацію моделі методом ковзного вікна.
        """
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
                seasonal_result = self.analyzer.detect_seasonality(data)
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

                # ВИПРАВЛЕННЯ: Покращена обробка MAPE з нульовими значеннями
                # Створюємо маску для ненульових значень
                mask = np.abs(test_data) > 1e-10
                nonzero_count = np.sum(mask)

                if nonzero_count > 0:
                    # Якщо є ненульові значення, розраховуємо MAPE на них
                    mape = np.mean(np.abs((test_data[mask] - forecast_values[mask]) / test_data[mask])) * 100
                    self.logger.info(
                        f"Ітерація {i + 1}: MAPE розраховано на {nonzero_count}/{len(test_data)} ненульових значеннях")
                else:
                    # Альтернативна метрика, якщо всі значення близькі до нуля
                    self.logger.warning(
                        f"Ітерація {i + 1}: Всі тестові значення близькі до нуля. Використовуємо альтернативну метрику.")
                    # Використовуємо MAE, нормалізований до середнього абсолютного значення прогнозу
                    denominator = max(np.mean(np.abs(forecast_values)), 1e-10)
                    mape = (mae / denominator) * 100

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
                            "mape": mape,
                            "nonzero_values": int(nonzero_count)  # Додаємо кількість ненульових значень
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
                            "mape": float(mape),
                            "nonzero_values": int(nonzero_count)  # Додаємо кількість ненульових значень
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
                            "nonzero_values": int(nonzero_count),  # Додаємо кількість ненульових значень
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
            Виконує аналіз залишків для заданої часової моделі, що зберігається у памʼяті або базі даних.

            Метод оцінює якість моделі через статистичні тести та автокореляційний аналіз залишків:
            - Статистика залишків (середнє, стандартне відхилення, мінімум, максимум, медіана)
            - Тест Джарка-Бера на нормальність розподілу залишків
            - Тест Льюнга-Бокса на автокореляцію
            - Тест Бройша-Паґана на гетероскедастичність
            - ACF та PACF аналіз
            - Перевірка, чи залишки можна вважати білим шумом

            Якщо дані не передано вручну, метод використовує дані, які використовувалися при навчанні моделі.

            Параметри:
            ----------
            model_key : str
                Ідентифікатор моделі, для якої виконується аналіз залишків.
            data : pd.Series, optional
                Часовий ряд, який слугує джерелом для аналізу залишків. Якщо не вказано, використовуються оригінальні дані моделі.

            Повертає:
            --------
            Dict
                Словник з результатами аналізу, включаючи:
                - статистики залишків
                - результати тестів Джарка-Бера, Льюнга-Бокса, Бройша-Паґана
                - ACF/PACF значення та значущі лаги
                - оцінку білого шуму
                - повідомлення про статус виконання
            """
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

    def compare_models(self, model_keys: List[str], test_data: pd.Series, window_size: int = None) -> Dict:
        """
            Порівнює кілька часових моделей за їхніми прогнозами на тестовому наборі даних.

            Метод генерує прогнози для кожної моделі, порівнює їх з фактичними даними,
            обчислює метрики якості (MSE, RMSE, MAE, MAPE) та волатильність, а також
            визначає найкращу модель за обраним критерієм (MSE за замовчуванням).

            Параметри
            ----------
            model_keys : List[str]
                Список ідентифікаторів моделей, які потрібно порівняти.
            test_data : pd.Series
                Часовий ряд із фактичними значеннями, що використовується для перевірки прогнозів моделей.
            window_size : int, optional
                Розмір вікна для обчислення волатильності. Якщо не задано, може використовуватись значення за замовчуванням.

            Повертає
            -------
            Dict
                Словник із результатами порівняння, включаючи:
                - статус та повідомлення
                - метрики для кожної моделі: MSE, RMSE, MAE, MAPE, волатильність
                - серії прогнозів
                - параметри моделей
                - інформацію про найкращу модель
                - (опціонально) збереження результатів у базі даних, якщо доступний db_manager


            """
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
                "mape": {},
                "volatility": {}
            }
        }

        try:
            # Розрахунок волатильності тестових даних для порівняння
            volatility_test = self._calculate_volatility(test_data, window_size)

            if volatility_test is not None:
                self.logger.info(f"Test data volatility: {volatility_test:.4f}")
            else:
                self.logger.warning("Could not calculate volatility for test data")

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

                    # Обчислення волатильності прогнозів з покращеною обробкою
                    volatility_pred = self._calculate_volatility(pred_series, window_size)

                    # Порівняння з волатильністю тестових даних
                    if volatility_test is not None and volatility_pred is not None:
                        volatility_diff = abs(volatility_pred - volatility_test)
                        volatility_ratio = volatility_pred / volatility_test if volatility_test != 0 else None
                    else:
                        volatility_diff = None
                        volatility_ratio = None

                    volatility_metrics = {
                        "predicted": float(volatility_pred) if volatility_pred is not None else None,
                        "actual": float(volatility_test) if volatility_test is not None else None,
                        "difference": float(volatility_diff) if volatility_diff is not None else None,
                        "ratio": float(volatility_ratio) if volatility_ratio is not None else None
                    }

                    comparison_results["metrics"]["mse"][model_key] = mse
                    comparison_results["metrics"]["rmse"][model_key] = rmse
                    comparison_results["metrics"]["mae"][model_key] = mae
                    comparison_results["metrics"]["mape"][model_key] = mape
                    comparison_results["metrics"]["volatility"][model_key] = volatility_metrics

                    comparison_results["models"][model_key] = {
                        "forecast": pred_series.to_dict(),
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "mape": mape,
                        "volatility": volatility_metrics,
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
                        "mape": comparison_results["metrics"]["mape"][best_model_key],
                        "volatility": comparison_results["metrics"]["volatility"][best_model_key]
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

    def calculate_volatility(self, data_series: pd.Series, window_size: int = None) -> float | None:
        """
        Розраховує волатильність часового ряду з правильною обробкою помилок.

        Args:
            data_series: Часовий ряд з фінансовими даними
            window_size: Розмір вікна для розрахунку ковзної волатільності.
                         Якщо None, розраховується для всієї серії.

        Returns:
            Значення волатільності або None у випадку помилки
        """
        if data_series is None or len(data_series) < 2:
            self.logger.warning("Insufficient data for volatility calculation")
            return None

        try:
            # Видалення пропущених значень
            clean_data = data_series.dropna()

            # Конвертація decimal.Decimal об'єктів у float
            if hasattr(clean_data.iloc[0], '__class__') and 'decimal' in str(type(clean_data.iloc[0])):
                clean_data = clean_data.astype(float)

            # Видалення нульових та від'ємних значень
            positive_data = clean_data[clean_data > 0]

            if len(positive_data) < 2:
                self.logger.warning("Insufficient positive values for volatility calculation")
                return None

            # Переконуємося, що дані у форматі Series
            if not isinstance(positive_data, pd.Series):
                positive_data = pd.Series(positive_data)

            # Розрахунок дохідності
            returns = positive_data / positive_data.shift(1)
            # Видалення NaN з першого запису
            returns = returns.dropna()

            if len(returns) < 1:
                self.logger.warning("No valid returns calculated for volatility")
                return None

            # Застосування логарифму до всієї серії
            # Переконуємося, що returns містить float значення
            returns = returns.astype(float)
            log_returns = np.log(returns)

            if window_size and window_size < len(log_returns):
                # Розрахунок ковзної волатільності
                self.logger.info(f"Calculating volatility with window size {window_size}")
                rolling_std = log_returns.rolling(window=window_size).std()
                # Використовуємо останнє значення ковзного стандартного відхилення
                volatility = rolling_std.iloc[-1] * np.sqrt(252)  # Річна волатільність
            else:
                # Розрахунок загальної волатільності
                self.logger.info("Calculating overall volatility")
                volatility = log_returns.std() * np.sqrt(252)  # Річна волатільність

            # Перевірка на NaN результат
            if pd.isna(volatility):
                self.logger.warning("Volatility calculation resulted in NaN")
                return None

            return float(volatility)

        except Exception as e:
            # Логування помилки і повернення None
            self.logger.error(f"Error during volatility calculation: {str(e)}")
            return None