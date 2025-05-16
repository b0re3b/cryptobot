# Файл arima_modeler.py
class ARIMAModeler:
    def __init__(self, logger, db_manager):
        self.logger = logger
        self.db_manager = db_manager
        self.models = {}

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