from datetime import datetime
from typing import Dict, Tuple, List

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
                # Виключаємо нульові значення при розрахунку MAPE
                mask = np.abs(y_true) > 1e-10  # Значення більше маленького порогу
                if np.any(mask):
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
                else:
                    self.logger.warning("All true values are close to zero, MAPE might be unreliable")
                    # Альтернативний розрахунок для запобігання ділення на нуль
                    epsilon = np.finfo(float).eps  # Дуже мале число
                    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
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

                # MAPE (виключаємо нульові і близькі до нуля значення)
                mask = np.abs(test_data) > 1e-10
                if np.any(mask):
                    mape = np.mean(np.abs((test_data[mask] - forecast_values[mask]) / test_data[mask])) * 100
                else:
                    # Альтернативний розрахунок для випадків з нульовими значеннями
                    mape = np.mean(np.abs((test_data - forecast_values) / (test_data + 1e-10))) * 100
                    self.logger.warning(
                        f"Ітерація {i + 1}: Всі тестові значення близькі до нуля. MAPE може бути ненадійним.")

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
                "mape": {},
                "volatility": {}  # Додаємо метрику волатильності
            }
        }

        try:
            # Розрахунок волатильності тестових даних для порівняння
            try:
                # FIX: Використовуємо np.log замість методу .log()
                test_returns = test_data.pct_change().dropna()

                if (test_returns <= -1).any():
                    self.logger.warning(
                        "Test data contains returns <= -100%, using absolute returns for volatility calculation")
                    # Використовуємо абсолютні значення змін для уникнення проблем з логарифмом
                    volatility_test = test_returns.std() * np.sqrt(252)  # Річна волатильність
                else:
                    # Безпечне обчислення логарифмічних прибутків
                    positive_values = test_data.dropna()
                    positive_values = positive_values[positive_values > 0]

                    if len(positive_values) > 1:
                        # FIX: Правильне використання np.log замість методу .log()
                        log_returns = np.log(positive_values / positive_values.shift(1)).dropna()
                        volatility_test = log_returns.std() * np.sqrt(252)  # Річна волатильність
                    else:
                        self.logger.warning("Not enough positive values for log-returns calculation")
                        volatility_test = test_returns.std() * np.sqrt(252)

                self.logger.info(f"Test data volatility: {volatility_test:.4f}")
            except Exception as vol_error:
                self.logger.error(f"Error during volatility calculation for test data: {str(vol_error)}")
                volatility_test = None

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

                    # Обчислення волатильності прогнозів
                    try:
                        # FIX: Використовуємо той самий виправлений підхід, що й для тестових даних
                        pred_returns = pred_series.pct_change().dropna()

                        if (pred_returns <= -1).any():
                            self.logger.warning(
                                f"Prediction for model {model_key} contains returns <= -100%, using absolute returns")
                            volatility_pred = pred_returns.std() * np.sqrt(252)
                        else:
                            # Безпечне обчислення логарифмічних прибутків
                            positive_pred = pred_series.dropna()
                            positive_pred = positive_pred[positive_pred > 0]

                            if len(positive_pred) > 1:
                                # FIX: Правильне використання np.log замість методу .log()
                                log_returns_pred = np.log(positive_pred / positive_pred.shift(1)).dropna()
                                volatility_pred = log_returns_pred.std() * np.sqrt(252)
                            else:
                                self.logger.warning(f"Not enough positive values for model {model_key}")
                                volatility_pred = pred_returns.std() * np.sqrt(252)

                        # Порівняння з волатильністю тестових даних
                        if volatility_test is not None:
                            volatility_diff = abs(volatility_pred - volatility_test)
                            volatility_ratio = volatility_pred / volatility_test if volatility_test != 0 else float('inf')
                        else:
                            volatility_diff = None
                            volatility_ratio = None

                        volatility_metrics = {
                            "predicted": float(volatility_pred),
                            "actual": float(volatility_test) if volatility_test is not None else None,
                            "difference": float(volatility_diff) if volatility_diff is not None else None,
                            "ratio": float(volatility_ratio) if volatility_ratio is not None else None
                        }
                    except Exception as vol_err:
                        self.logger.error(f"Error calculating volatility for model {model_key}: {str(vol_err)}")
                        volatility_metrics = {"error": str(vol_err)}

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