# Файл forecaster.py
class Forecaster:
    def __init__(self, logger, db_manager):
        self.logger = logger
        self.db_manager = db_manager

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