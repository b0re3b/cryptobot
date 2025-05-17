from datetime import datetime
from typing import Tuple, Dict,Any

import numpy as np
import pandas as pd
from pmdarima import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os


class ARIMAModeler:
    """
    Клас для створення, навчання та збереження моделей ARIMA та SARIMA.

    Дозволяє виконувати:
    - Підгонку моделей ARIMA/SARIMA до часових рядів
    - Збереження моделей в пам'яті та на диску
    - Завантаження моделей з диску або бази даних
    """

    def __init__(self, logger, db_manager=None):
        """
        Ініціалізація класу ARIMAModeler.

        Параметри:
        ----------
        logger : об'єкт для логування
            Об'єкт, що підтримує методи info, warning, error для логування
        db_manager : об'єкт, optional
            Об'єкт для взаємодії з базою даних. За замовчуванням None.
        """
        self.logger = logger
        self.db_manager = db_manager
        self.models = {}
        self.transformations = {}  # Додано відсутній атрибут для трансформацій

    def _validate_data(self, data: pd.Series, min_required: int) -> pd.Series:
        """
        Валідація вхідних даних перед підгонкою моделі.

        Параметри:
        ----------
        data : pd.Series
            Вхідні дані для моделі
        min_required : int
            Мінімальна необхідна кількість спостережень

        Повертає:
        ---------
        pd.Series
            Валідовані дані

        Викидає:
        --------
        ValueError
            Якщо дані не відповідають вимогам
        """
        # Перевірка на NaN значення
        if data.isnull().any():
            self.logger.warning("Дані містять NaN значення. Видаляємо їх перед підгонкою.")
            data = data.dropna()

        # Перевірка достатньої кількості даних
        if len(data) < min_required:
            error_msg = f"Недостатньо точок даних для моделі. Потрібно мінімум {min_required}, маємо {len(data)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return data

    def _generate_model_key(self, model_type: str, symbol: str) -> str:
        """
        Генерує унікальний ключ для моделі.

        Параметри:
        ----------
        model_type : str
            Тип моделі ('arima' або 'sarima')
        symbol : str
            Символ або ідентифікатор набору даних

        Повертає:
        ---------
        str
            Унікальний ключ моделі
        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{symbol}_{model_type}_{timestamp}"

    def _save_model_to_db(self, model_key: str, model_info: Dict) -> bool:
        """
        Зберігає модель у базу даних.

        Параметри:
        ----------
        model_key : str
            Ключ моделі
        model_info : Dict
            Інформація про модель

        Повертає:
        ---------
        bool
            True, якщо збереження успішне, інакше False
        """
        if self.db_manager is None:
            return False

        try:
            # Зберігаємо метадані
            self.db_manager.save_model_metadata(model_key, model_info["metadata"])

            # Зберігаємо параметри
            self.db_manager.save_model_parameters(model_key, model_info["parameters"])

            # Зберігаємо двійкове представлення моделі
            model_binary = pickle.dumps(model_info["fit_result"])
            self.db_manager.save_model_binary(model_key, model_binary)

            # Якщо є трансформації, зберігаємо їх також
            if model_key in self.transformations:
                self.db_manager.save_data_transformations(model_key, self.transformations[model_key])

            self.logger.info(f"Модель {model_key} збережена в базу даних")
            return True
        except Exception as db_error:
            self.logger.error(f"Помилка збереження моделі в базу даних: {str(db_error)}")
            return False

    def _collect_model_metadata(self, data: pd.Series, model_key: str, model_type: str,
                                symbol: str) -> Dict:
        """
        Збирає метадані моделі.

        Параметри:
        ----------
        data : pd.Series
            Дані, використані для підгонки моделі
        model_key : str
            Ключ моделі
        model_type : str
            Тип моделі ('ARIMA' або 'SARIMA')
        symbol : str
            Символ або ідентифікатор набору даних

        Повертає:
        ---------
        Dict
            Словник з метаданими моделі
        """
        start_date = data.index[0]
        end_date = data.index[-1]

        # Конвертація дат в строки для серіалізації
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        else:
            start_date = str(start_date)

        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        else:
            end_date = str(end_date)

        return {
            "model_type": model_type,
            "symbol": symbol,
            "timestamp": datetime.now(),
            "data_range": {
                "start": start_date,
                "end": end_date,
                "length": len(data)
            },
            "model_key": model_key
        }

    def _create_model_info(self, fit_result: Any, data: pd.Series, model_key: str,
                           model_type: str, symbol: str, params: Dict) -> Dict:
        """
        Створює повну інформацію про модель.

        Параметри:
        ----------
        fit_result : Any
            Результат підгонки моделі
        data : pd.Series
            Дані, використані для підгонки
        model_key : str
            Ключ моделі
        model_type : str
            Тип моделі ('ARIMA' або 'SARIMA')
        symbol : str
            Символ або ідентифікатор набору даних
        params : Dict
            Параметри моделі

        Повертає:
        ---------
        Dict
            Повна інформація про модель
        """
        # Збираємо метадані
        metadata = self._collect_model_metadata(data, model_key, model_type, symbol)

        # Додаємо інформацію про збіжність
        training_info = {
            "convergence": True if fit_result.mle_retvals.get('converged', False) else False,
            "iterations": fit_result.mle_retvals.get('iterations', None)
        }

        # Додаємо параметри до загальних параметрів
        params["training_info"] = training_info

        # Збираємо статистику моделі
        stats = {
            "aic": fit_result.aic,
            "bic": fit_result.bic,
            "aicc": fit_result.aicc if hasattr(fit_result, 'aicc') else None,
            "log_likelihood": fit_result.llf
        }

        return {
            "model": None,  # Саму модель не зберігаємо, щоб зменшити використання пам'яті
            "fit_result": fit_result,
            "metadata": metadata,
            "parameters": params,
            "stats": stats
        }

    def fit_arima(self, data: pd.Series, order: Tuple[int, int, int],
                  symbol: str = 'default') -> Dict:
        """
        Підгонка моделі ARIMA до часового ряду.

        Параметри:
        ----------
        data : pd.Series
            Часовий ряд для моделювання
        order : Tuple[int, int, int]
            Порядок моделі ARIMA (p, d, q)
        symbol : str, optional
            Ідентифікатор набору даних. За замовчуванням 'default'.

        Повертає:
        ---------
        Dict
            Результат підгонки моделі
        """
        self.logger.info(f"Починаємо навчання моделі ARIMA з порядком {order} для символу {symbol}")

        try:
            # Валідація даних
            min_required = sum(order) + 1
            data = self._validate_data(data, min_required)

            # Генеруємо ключ моделі
            model_key = self._generate_model_key("arima", symbol)

            # Створюємо та навчаємо модель
            model = ARIMA(data, order=order)
            fit_result = model.fit()

            # Формуємо параметри моделі
            params = {"order": order}

            # Створюємо повну інформацію про модель
            model_info = self._create_model_info(
                fit_result, data, model_key, "ARIMA", symbol, params
            )

            # Зберігаємо модель в словнику
            self.models[model_key] = model_info

            # Зберігаємо модель в БД, якщо доступно
            self._save_model_to_db(model_key, model_info)

            self.logger.info(f"Модель ARIMA {model_key} успішно навчена")

            return {
                "status": "success",
                "message": "Модель ARIMA успішно навчена",
                "model_key": model_key,
                "model_info": {
                    "metadata": model_info["metadata"],
                    "parameters": model_info["parameters"],
                    "stats": model_info["stats"]
                }
            }

        except ValueError as ve:
            # Повертаємо помилку валідації
            return {
                "status": "error",
                "message": str(ve),
                "model_key": None,
                "model_info": None
            }

        except Exception as e:
            self.logger.error(f"Помилка під час навчання моделі ARIMA: {str(e)}")
            return {
                "status": "error",
                "message": f"Помилка під час навчання моделі ARIMA: {str(e)}",
                "model_key": None,
                "model_info": None
            }

    def fit_sarima(self, data: pd.Series, order: Tuple[int, int, int],
                   seasonal_order: Tuple[int, int, int, int], symbol: str = 'default') -> Dict:
        """
        Підгонка моделі SARIMA до часового ряду.

        Параметри:
        ----------
        data : pd.Series
            Часовий ряд для моделювання
        order : Tuple[int, int, int]
            Порядок моделі ARIMA (p, d, q)
        seasonal_order : Tuple[int, int, int, int]
            Сезонний порядок моделі (P, D, Q, s)
        symbol : str, optional
            Ідентифікатор набору даних. За замовчуванням 'default'.

        Повертає:
        ---------
        Dict
            Результат підгонки моделі
        """
        self.logger.info(
            f"Починаємо навчання моделі SARIMA з порядком {order}, сезонним порядком {seasonal_order} для символу {symbol}")

        try:
            # Валідація даних
            min_required = sum(order) + sum(seasonal_order[:-1]) + 2 * seasonal_order[-1]
            data = self._validate_data(data, min_required)

            # Генеруємо ключ моделі
            model_key = self._generate_model_key("sarima", symbol)

            # Створюємо та навчаємо модель
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

            # Формуємо параметри моделі
            params = {
                "order": order,
                "seasonal_order": seasonal_order
            }

            # Створюємо повну інформацію про модель
            model_info = self._create_model_info(
                fit_result, data, model_key, "SARIMA", symbol, params
            )

            # Зберігаємо модель в словнику
            self.models[model_key] = model_info

            # Зберігаємо модель в БД, якщо доступно
            self._save_model_to_db(model_key, model_info)

            self.logger.info(f"Модель SARIMA {model_key} успішно навчена")

            return {
                "status": "success",
                "message": "Модель SARIMA успішно навчена",
                "model_key": model_key,
                "model_info": {
                    "metadata": model_info["metadata"],
                    "parameters": model_info["parameters"],
                    "stats": model_info["stats"]
                }
            }

        except ValueError as ve:
            # Повертаємо помилку валідації
            return {
                "status": "error",
                "message": str(ve),
                "model_key": None,
                "model_info": None
            }

        except Exception as e:
            self.logger.error(f"Помилка під час навчання моделі SARIMA: {str(e)}")
            return {
                "status": "error",
                "message": f"Помилка під час навчання моделі SARIMA: {str(e)}",
                "model_key": None,
                "model_info": None
            }

    def get_model_forecast(self, model_key: str, steps: int,
                           return_conf_int: bool = True, alpha: float = 0.05) -> Dict:
        """
        Отримує прогноз моделі на вказану кількість кроків.

        Параметри:
        ----------
        model_key : str
            Ключ моделі
        steps : int
            Кількість кроків для прогнозування
        return_conf_int : bool, optional
            Чи повертати інтервали довіри. За замовчуванням True.
        alpha : float, optional
            Рівень значущості для інтервалів довіри. За замовчуванням 0.05 (95% інтервал).

        Повертає:
        ---------
        Dict
            Результат прогнозування
        """
        self.logger.info(f"Отримання прогнозу для моделі {model_key} на {steps} кроків")

        # Перевіряємо наявність моделі в пам'яті
        if model_key not in self.models:
            # Якщо моделі немає в пам'яті, спробуємо завантажити її з БД
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Модель {model_key} не знайдена в пам'яті, спроба завантаження з БД")
                    loaded = self.db_manager.load_complete_model(model_key)
                    if not loaded:
                        error_msg = f"Не вдалося завантажити модель {model_key} з бази даних"
                        self.logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
                except Exception as e:
                    error_msg = f"Помилка завантаження моделі {model_key} з бази даних: {str(e)}"
                    self.logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
            else:
                error_msg = f"Модель {model_key} не знайдена і не надано менеджер бази даних"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

        try:
            # Отримуємо модель
            model_info = self.models[model_key]
            fit_result = model_info["fit_result"]

            # Отримуємо прогноз
            if return_conf_int:
                forecast, conf_int = fit_result.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
                result = {
                    "status": "success",
                    "forecast": forecast.tolist() if isinstance(forecast, np.ndarray) else forecast,
                    "conf_int_lower": conf_int[:, 0].tolist() if isinstance(conf_int, np.ndarray) else conf_int[0],
                    "conf_int_upper": conf_int[:, 1].tolist() if isinstance(conf_int, np.ndarray) else conf_int[1],
                    "alpha": alpha
                }
            else:
                forecast = fit_result.predict(n_periods=steps, return_conf_int=False)
                result = {
                    "status": "success",
                    "forecast": forecast.tolist() if isinstance(forecast, np.ndarray) else forecast
                }

            self.logger.info(f"Прогноз для моделі {model_key} успішно отриманий")
            return result

        except Exception as e:
            error_msg = f"Помилка отримання прогнозу: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def save_model(self, model_key: str, path: str) -> bool:
        """
        Зберігає модель у файл на диску.

        Параметри:
        ----------
        model_key : str
            Ключ моделі
        path : str
            Шлях для збереження моделі

        Повертає:
        ---------
        bool
            True, якщо збереження успішне, інакше False
        """
        self.logger.info(f"Починаємо зберігати модель {model_key} у {path}")

        # Перевіряємо наявність моделі в пам'яті
        if model_key not in self.models:
            # Якщо моделі немає в пам'яті, спробуємо завантажити її з БД
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Модель {model_key} не знайдена в пам'яті, спроба завантаження з БД")
                    loaded = self.db_manager.load_complete_model(model_key)
                    if not loaded:
                        error_msg = f"Не вдалося завантажити модель {model_key} з бази даних"
                        self.logger.error(error_msg)
                        return False
                except Exception as e:
                    error_msg = f"Помилка завантаження моделі {model_key} з бази даних: {str(e)}"
                    self.logger.error(error_msg)
                    return False
            else:
                error_msg = f"Модель {model_key} не знайдена і не надано менеджер бази даних"
                self.logger.error(error_msg)
                return False

        try:
            # Отримуємо модель
            model_info = self.models[model_key]

            # Перевіряємо наявність необхідних компонентів
            if "fit_result" not in model_info:
                error_msg = f"Модель {model_key} не має результату підгонки"
                self.logger.error(error_msg)
                return False

            # Створюємо директорію, якщо вона не існує
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"Створено директорію: {directory}")

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

            self.logger.info(f"Модель {model_key} успішно збережена у {path}")
            return True

        except Exception as e:
            error_msg = f"Помилка збереження моделі на диск: {str(e)}"
            self.logger.error(error_msg)
            return False

    def load_model(self, model_key: str, path: str) -> bool:
        """
        Завантажує модель з файлу на диску.

        Параметри:
        ----------
        model_key : str
            Ключ моделі
        path : str
            Шлях до файлу моделі

        Повертає:
        ---------
        bool
            True, якщо завантаження успішне, інакше False
        """
        self.logger.info(f"Завантаження моделі з шляху: {path}")

        try:
            # Перевірка існування файлу
            if not os.path.exists(path):
                self.logger.error(f"Файл моделі не знайдено за шляхом: {path}")
                return False

            # Завантаження моделі з файлу
            with open(path, 'rb') as file:
                loaded_data = pickle.load(file)

            # Перевірка структури завантажених даних
            required_keys = ["fit_result", "metadata", "parameters", "stats"]
            if not all(key in loaded_data for key in required_keys):
                self.logger.error("Завантажені дані моделі мають неправильну структуру")
                return False

            # Записуємо модель у внутрішній словник
            self.models[model_key] = loaded_data

            # Оновлюємо ключ моделі в метаданих, якщо він відрізняється
            self.models[model_key]["metadata"]["model_key"] = model_key

            # Зберігаємо трансформації, якщо є
            if "transformations" in loaded_data:
                self.transformations[model_key] = loaded_data["transformations"]

            # Зберігаємо модель в БД, якщо доступний менеджер БД
            if self.db_manager is not None:
                try:
                    # Створюємо бінарне представлення моделі
                    model_binary = pickle.dumps(loaded_data["fit_result"])

                    # Зберігаємо всі компоненти моделі
                    self.db_manager.save_model_metadata(model_key, loaded_data["metadata"])
                    self.db_manager.save_model_parameters(model_key, loaded_data["parameters"])
                    self.db_manager.save_model_binary(model_key, model_binary)

                    # Зберігаємо метрики, якщо є
                    if "stats" in loaded_data:
                        self.db_manager.save_model_metrics(model_key, loaded_data["stats"])

                    # Зберігаємо трансформації, якщо є
                    if "transformations" in loaded_data:
                        self.db_manager.save_data_transformations(model_key, loaded_data["transformations"])

                    self.logger.info(f"Модель {model_key} збережена в базу даних після завантаження з файлу")
                except Exception as db_error:
                    self.logger.error(f"Помилка збереження завантаженої моделі в базу даних: {str(db_error)}")

            self.logger.info(f"Модель {model_key} успішно завантажена з {path}")
            return True

        except Exception as e:
            self.logger.error(f"Помилка завантаження моделі з файлу: {str(e)}")
            return False

    def add_data_transformation(self, model_key: str, transformation_info: Dict) -> bool:
        """
        Додає інформацію про трансформацію даних для моделі.

        Параметри:
        ----------
        model_key : str
            Ключ моделі
        transformation_info : Dict
            Інформація про трансформацію

        Повертає:
        ---------
        bool
            True, якщо додавання успішне, інакше False
        """
        try:
            if model_key not in self.transformations:
                self.transformations[model_key] = {}

            # Додаємо нову трансформацію
            transformation_id = f"transform_{len(self.transformations[model_key]) + 1}"
            self.transformations[model_key][transformation_id] = transformation_info

            # Зберігаємо в БД, якщо доступно
            if self.db_manager is not None:
                try:
                    self.db_manager.save_data_transformations(model_key, self.transformations[model_key])
                except Exception as db_error:
                    self.logger.error(f"Помилка збереження трансформацій в БД: {str(db_error)}")

            return True
        except Exception as e:
            self.logger.error(f"Помилка додавання трансформації: {str(e)}")
            return False