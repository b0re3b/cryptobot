from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os
import traceback
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from utils.logger import CryptoLogger
from data.db import DatabaseManager
from timeseriesmodels.TimeSeriesAnalyzer import TimeSeriesAnalyzer


class ARIMAModeler:

    def __init__(self):
        """
        Ініціалізація класу ARIMAModeler.
        """
        self.logger = CryptoLogger('ArimaModeler')
        self.db_manager = DatabaseManager()
        self.models = {}
        self.transformations = {}
        self.ts_analyzer = TimeSeriesAnalyzer()

        # Логування ініціалізації
        self.logger.info("ARIMAModeler ініціалізовано")
        self.logger.debug(f"База даних підключена: {self.db_manager is not None}")
        self.logger.debug(f"Аналізатор часових рядів ініціалізовано: {self.ts_analyzer is not None}")

    def _validate_data(self, data: pd.Series, min_required: int) -> pd.Series:

        self.logger.debug(
            f"Початок валідації даних. Тип: {type(data)}, розмір: {len(data) if hasattr(data, '__len__') else 'невідомо'}")

        # Перевіряємо тип даних
        if not isinstance(data, pd.Series):
            self.logger.info("Конвертація даних в pandas Series")
            try:
                data = pd.Series(data)
                self.logger.debug(f"Дані успішно конвертовані в Series розміром {len(data)}")
            except Exception as e:
                error_msg = f"Не вдалося конвертувати дані в pandas Series: {str(e)}"
                self.logger.error(error_msg)
                self.logger.debug(f"Стек помилки: {traceback.format_exc()}")
                raise ValueError(error_msg)

        # Логування інформації про вхідні дані
        self.logger.info(f"Валідація даних: розмір={len(data)}, тип даних={data.dtype}, мін. потрібно={min_required}")

        # Перевіряємо тип даних значень - повинні бути числові
        if data.dtype == 'object':
            self.logger.warning("Дані мають тип 'object', спроба конвертації в числовий тип")
            try:
                data = pd.to_numeric(data, errors='coerce')
                self.logger.info(f"Дані успішно конвертовані в числовий тип: {data.dtype}")
            except Exception as e:
                error_msg = f"Дані містять нечислові значення: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Перевірка на NaN значення
        nan_count = data.isnull().sum()
        if nan_count > 0:
            self.logger.warning(
                f"Знайдено {nan_count} NaN значень з {len(data)} загальних точок ({nan_count / len(data) * 100:.2f}%)")
            data_before = len(data)
            data = data.dropna()
            data_after = len(data)
            self.logger.info(f"Видалено {data_before - data_after} NaN значень. Залишилось {data_after} точок")

        # Перевірка достатньої кількості даних
        if len(data) < min_required:
            error_msg = f"Недостатньо точок даних для моделі. Потрібно мінімум {min_required}, маємо {len(data)}"
            self.logger.error(error_msg)
            self.logger.debug(
                f"Статистика даних: мін={data.min():.4f}, макс={data.max():.4f}, середнє={data.mean():.4f}")
            raise ValueError(error_msg)

        # Перевірка на константні значення
        unique_values = data.nunique()
        if unique_values <= 1:
            error_msg = f"Дані містять тільки {unique_values} унікальних значень, неможливо навчити модель"
            self.logger.error(error_msg)
            if len(data) > 0:
                self.logger.debug(f"Значення в даних: {data.unique()[:10]}...")  # Показати перші 10 унікальних значень
            raise ValueError(error_msg)

        # Логування статистики валідованих даних
        self.logger.info(f"Валідація успішна. Фінальний розмір: {len(data)}, унікальних значень: {unique_values}")
        self.logger.debug(
            f"Статистика: мін={data.min():.4f}, макс={data.max():.4f}, середнє={data.mean():.4f}, стд={data.std():.4f}")

        return data

    def _generate_model_key(self, model_type: str, symbol: str) -> str:

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_key = f"{symbol}_{model_type}_{timestamp}"
        self.logger.debug(f"Згенеровано ключ моделі: {model_key}")
        return model_key

    def _save_model_to_db(self, model_key: str, model_info: Dict) -> bool:

        if self.db_manager is None:
            self.logger.warning("База даних недоступна, пропускаємо збереження моделі")
            return False

        self.logger.info(f"Початок збереження моделі {model_key} в базу даних")

        try:
            metadata = model_info["metadata"]
            self.logger.debug(f"Метадані моделі: {list(metadata.keys())}")

            # Витягуємо необхідні параметри для збереження
            model_type = metadata.get("model_type", "UNKNOWN")
            timeframe = metadata.get("timeframe", "1d")

            self.logger.debug(f"Тип моделі: {model_type}, часовий період: {timeframe}")

            # Отримуємо дати з data_range
            data_range = metadata.get("data_range", {})
            start_date = data_range.get("start", datetime.now().isoformat())
            end_date = data_range.get("end", datetime.now().isoformat())
            data_length = data_range.get("length", 0)

            self.logger.debug(f"Діапазон даних: {start_date} - {end_date}, точок даних: {data_length}")

            # Конвертуємо дати в datetime об'єкти, якщо вони в форматі строки
            if isinstance(start_date, str):
                try:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    self.logger.debug("Дата початку успішно конвертована")
                except Exception as e:
                    self.logger.warning(f"Не вдалося конвертувати дату початку: {e}")
                    start_date = datetime.now()

            if isinstance(end_date, str):
                try:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    self.logger.debug("Дата кінця успішно конвертована")
                except Exception as e:
                    self.logger.warning(f"Не вдалося конвертувати дату кінця: {e}")
                    end_date = datetime.now()

            # Зберігаємо метадані з усіма необхідними параметрами
            self.logger.debug("Збереження метаданих моделі")
            self.db_manager.save_model_metadata(
                model_key,
                model_type,
                timeframe,
                start_date,
                end_date,
                metadata
            )

            # Зберігаємо параметри
            self.logger.debug(f"Збереження параметрів моделі: {list(model_info['parameters'].keys())}")
            self.db_manager.save_model_parameters(model_key, model_info["parameters"])

            # Зберігаємо двійкове представлення моделі
            self.logger.debug("Серіалізація та збереження двійкового представлення моделі")
            model_binary = pickle.dumps(model_info["fit_result"])
            binary_size = len(model_binary)
            self.logger.debug(f"Розмір серіалізованої моделі: {binary_size} байт")
            self.db_manager.save_model_binary(model_key, model_binary)

            # Якщо є трансформації, зберігаємо їх також
            if model_key in self.transformations:
                self.logger.debug("Збереження трансформацій даних")
                self.db_manager.save_data_transformations(model_key, self.transformations[model_key])
            else:
                self.logger.debug("Трансформації даних відсутні")

            self.logger.info(f"Модель {model_key} успішно збережена в базу даних")
            return True

        except Exception as db_error:
            error_msg = f"Помилка збереження моделі в базу даних: {str(db_error)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Повний стек помилки: {traceback.format_exc()}")
            return False

    def _collect_model_metadata(self, data: pd.Series, model_key: str, model_type: str,
                                symbol: str, timeframe: str = "1d") -> Dict:

        self.logger.debug(f"Збирання метаданих для моделі {model_key}")

        start_date = data.index[0] if len(data.index) > 0 else datetime.now()
        end_date = data.index[-1] if len(data.index) > 0 else datetime.now()

        self.logger.debug(f"Діапазон даних: {start_date} - {end_date}")

        # Конвертація дат в строки для серіалізації
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        else:
            start_date = str(start_date)

        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        else:
            end_date = str(end_date)

        metadata = {
            "model_type": model_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now(),
            "data_range": {
                "start": start_date,
                "end": end_date,
                "length": len(data)
            },
            "model_key": model_key
        }

        self.logger.debug(f"Метадані зібрані: тип={model_type}, символ={symbol}, точок даних={len(data)}")
        return metadata

    def _create_model_info(self, fit_result: Any, data: pd.Series, model_key: str,
                           model_type: str, symbol: str, params: Dict, timeframe: str = "1d") -> Dict:

        self.logger.debug(f"Створення інформації про модель {model_key}")

        # Збираємо метадані
        metadata = self._collect_model_metadata(data, model_key, model_type, symbol, timeframe)

        # Додаємо інформацію про збіжність
        convergence_info = self._extract_convergence_info(fit_result)
        self.logger.debug(f"Інформація про збіжність: {convergence_info}")

        training_info = {
            "convergence": convergence_info.get('converged', False),
            "iterations": convergence_info.get('iterations', None),
            "final_log_likelihood": convergence_info.get('final_llf', None)
        }

        # Додаємо параметри до загальних параметрів
        params["training_info"] = training_info

        # Збираємо статистику моделі
        stats = self._extract_model_stats(fit_result)
        self.logger.info(f"Статистика моделі {model_key}: AIC={stats.get('aic', 'N/A')}, BIC={stats.get('bic', 'N/A')}")

        model_info = {
            "model": None,  # Саму модель не зберігаємо, щоб зменшити використання пам'яті
            "fit_result": fit_result,
            "metadata": metadata,
            "parameters": params,
            "stats": stats
        }

        self.logger.debug("Інформація про модель успішно створена")
        return model_info

    def _extract_convergence_info(self, fit_result: Any) -> Dict:
        """Витягує інформацію про збіжність з результату підгонки."""
        convergence_info = {}

        try:
            if hasattr(fit_result, 'mle_retvals'):
                mle_retvals = fit_result.mle_retvals
                convergence_info['converged'] = mle_retvals.get('converged', False)
                convergence_info['iterations'] = mle_retvals.get('iterations', None)
                convergence_info['final_llf'] = mle_retvals.get('fopt', None)
                self.logger.debug(
                    f"MLE збіжність: {convergence_info['converged']}, ітерацій: {convergence_info['iterations']}")
            else:
                convergence_info['converged'] = True  # Припускаємо збіжність якщо немає інформації
                self.logger.debug("Інформація про MLE збіжність недоступна")

        except Exception as e:
            self.logger.warning(f"Не вдалося витягти інформацію про збіжність: {e}")
            convergence_info['converged'] = False

        return convergence_info

    def _extract_model_stats(self, fit_result: Any) -> Dict:
        """Витягує статистику моделі з результату підгонки."""
        stats = {}

        stat_attrs = ['aic', 'bic', 'aicc', 'llf', 'hqic']
        for attr in stat_attrs:
            try:
                value = getattr(fit_result, attr, None)
                stats[attr] = value
                if value is not None:
                    self.logger.debug(f"Статистика {attr.upper()}: {value:.4f}")
            except Exception as e:
                self.logger.debug(f"Не вдалося отримати статистику {attr}: {e}")
                stats[attr] = None

        return stats

    def _extract_optimal_params(self, optimal_params: Dict, is_seasonal: bool = False) -> Tuple[
        Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Витягує оптимальні параметри ARIMA/SARIMA з результату аналізу.

        Args:
            optimal_params: Словник з результатами оптимізації
            is_seasonal: True для SARIMA, False для ARIMA

        Returns:
            Tuple[order, seasonal_order] для SARIMA або Tuple[order, None] для ARIMA
        """
        self.logger.debug(f"Витягування оптимальних параметрів. Сезонна модель: {is_seasonal}")
        self.logger.debug(f"Структура вхідних параметрів: {list(optimal_params.keys())}")

        # Стандартні параметри за замовчуванням
        default_order = (1, 1, 1)
        default_seasonal = (1, 1, 1, 7) if is_seasonal else None

        try:
            # Основний алгоритм витягування параметрів
            extracted_order, extracted_seasonal = self._parse_parameters_structure(optimal_params, is_seasonal)

            if extracted_order is not None:
                # Валідація параметрів order
                if self._validate_order_params(extracted_order):
                    seasonal_result = extracted_seasonal if is_seasonal else None
                    self.logger.info(
                        f"Успішно витягнуто параметри: order={extracted_order}, "
                        f"seasonal_order={seasonal_result}"
                    )
                    return extracted_order, seasonal_result
                else:
                    self.logger.warning(f"Невалідні параметри order: {extracted_order}")

        except Exception as e:
            self.logger.error(f"Помилка під час витягування параметрів: {str(e)}")
            self.logger.debug(f"Стек помилки: {traceback.format_exc()}")

        # Детальне логування для діагностики
        self._log_structure_for_diagnosis(optimal_params)

        # Повертаємо стандартні параметри
        self.logger.warning(
            f"Не вдалося витягти параметри. Використовуємо стандартні: "
            f"order={default_order}, seasonal_order={default_seasonal}"
        )
        return default_order, default_seasonal

    def _parse_parameters_structure(self, optimal_params: Dict, is_seasonal: bool) -> Tuple[
        Optional[Tuple], Optional[Tuple]]:
        """
        Парсить різні можливі структури параметрів.

        Returns:
            Tuple[order, seasonal_order] або Tuple[None, None] якщо не знайдено
        """

        # Стратегії пошуку параметрів (у порядку пріоритету)
        search_strategies = [
            self._extract_from_parameters_section,
            self._extract_from_model_info_section,
            self._extract_from_top_level,
            self._extract_from_individual_params,
            self._extract_from_model_attributes,
            self._extract_from_nested_dictionaries
        ]

        for strategy in search_strategies:
            try:
                order, seasonal_order = strategy(optimal_params, is_seasonal)
                if order is not None:
                    return order, seasonal_order
            except Exception as e:
                self.logger.debug(f"Стратегія {strategy.__name__} не спрацювала: {str(e)}")
                continue

        return None, None

    def _extract_from_parameters_section(self, optimal_params: Dict, is_seasonal: bool) -> Tuple[
        Optional[Tuple], Optional[Tuple]]:
        """Витягує параметри з секції 'parameters'."""
        if 'parameters' not in optimal_params:
            return None, None

        params = optimal_params['parameters']
        self.logger.debug(f"Знайдено секцію 'parameters': {params}")

        if not isinstance(params, dict):
            return None, None

        # Пряме витягування order та seasonal_order
        if 'order' in params:
            order = self._ensure_tuple(params['order'])
            seasonal_order = None

            if is_seasonal and 'seasonal_order' in params:
                seasonal_order = self._ensure_tuple(params['seasonal_order'])
            elif is_seasonal:
                # Спробувати створити seasonal_order з окремих параметрів
                seasonal_order = self._build_seasonal_order_from_params(params)

            return order, seasonal_order

        # Витягування з окремих параметрів p, d, q
        if all(key in params for key in ['p', 'd', 'q']):
            order = (params['p'], params['d'], params['q'])
            seasonal_order = None

            if is_seasonal:
                seasonal_order = (
                    params.get('P', 0),
                    params.get('D', 0),
                    params.get('Q', 0),
                    params.get('s', 7)
                )

            return order, seasonal_order

        return None, None

    def _extract_from_model_info_section(self, optimal_params: Dict, is_seasonal: bool) -> Tuple[
        Optional[Tuple], Optional[Tuple]]:
        """Витягує параметри з секції 'model_info'."""
        if 'model_info' not in optimal_params:
            return None, None

        model_info = optimal_params['model_info']
        self.logger.debug(f"Знайдено секцію 'model_info': {type(model_info)}")

        if isinstance(model_info, dict):
            # Пряме витягування з model_info
            if 'order' in model_info:
                order = self._ensure_tuple(model_info['order'])
                seasonal_order = None

                if is_seasonal and 'seasonal_order' in model_info:
                    seasonal_order = self._ensure_tuple(model_info['seasonal_order'])
                elif is_seasonal:
                    seasonal_order = self._build_seasonal_order_from_params(model_info)

                return order, seasonal_order

            # Вкладені параметри в model_info
            if 'parameters' in model_info:
                return self._extract_from_parameters_section({'parameters': model_info['parameters']}, is_seasonal)

            # Окремі параметри в model_info
            if all(key in model_info for key in ['p', 'd', 'q']):
                order = (model_info['p'], model_info['d'], model_info['q'])
                seasonal_order = None

                if is_seasonal:
                    seasonal_order = (
                        model_info.get('P', 0),
                        model_info.get('D', 0),
                        model_info.get('Q', 0),
                        model_info.get('s', 7)
                    )

                return order, seasonal_order

        # Об'єкт моделі з атрибутами
        if hasattr(model_info, 'order'):
            order = model_info.order
            seasonal_order = None

            if is_seasonal:
                seasonal_order = getattr(model_info, 'seasonal_order', (0, 0, 0, 7))

            return order, seasonal_order

        return None, None

    def _extract_from_top_level(self, optimal_params: Dict, is_seasonal: bool) -> Tuple[
        Optional[Tuple], Optional[Tuple]]:
        """Витягує параметри з верхнього рівня словника."""
        if 'order' in optimal_params:
            order = self._ensure_tuple(optimal_params['order'])
            seasonal_order = None

            if is_seasonal and 'seasonal_order' in optimal_params:
                seasonal_order = self._ensure_tuple(optimal_params['seasonal_order'])
            elif is_seasonal:
                seasonal_order = self._build_seasonal_order_from_params(optimal_params)

            return order, seasonal_order

        return None, None

    def _extract_from_individual_params(self, optimal_params: Dict, is_seasonal: bool) -> Tuple[
        Optional[Tuple], Optional[Tuple]]:
        """Витягує параметри з окремих ключів p, d, q."""
        if all(key in optimal_params for key in ['p', 'd', 'q']):
            order = (optimal_params['p'], optimal_params['d'], optimal_params['q'])
            seasonal_order = None

            if is_seasonal:
                seasonal_order = (
                    optimal_params.get('P', 0),
                    optimal_params.get('D', 0),
                    optimal_params.get('Q', 0),
                    optimal_params.get('s', 7)
                )

            return order, seasonal_order

        return None, None

    def _extract_from_model_attributes(self, optimal_params: Dict, is_seasonal: bool) -> Tuple[
        Optional[Tuple], Optional[Tuple]]:
        """Витягує параметри з атрибутів об'єкта моделі."""
        # Якщо результат це сама модель з атрибутом order
        if hasattr(optimal_params, 'order'):
            order = optimal_params.order
            seasonal_order = None

            if is_seasonal:
                seasonal_order = getattr(optimal_params, 'seasonal_order', (0, 0, 0, 7))

            return order, seasonal_order

        # Перевіряємо вкладений ключ 'model'
        if 'model' in optimal_params and hasattr(optimal_params['model'], 'order'):
            order = optimal_params['model'].order
            seasonal_order = None

            if is_seasonal:
                seasonal_order = getattr(optimal_params['model'], 'seasonal_order', (0, 0, 0, 7))

            return order, seasonal_order

        # Перевіряємо структуру результату auto_arima
        if 'arima_model' in optimal_params and hasattr(optimal_params['arima_model'], 'order'):
            arima_model = optimal_params['arima_model']
            order = arima_model.order
            seasonal_order = None

            if is_seasonal:
                seasonal_order = getattr(arima_model, 'seasonal_order', (0, 0, 0, 7))

            return order, seasonal_order

        return None, None

    def _extract_from_nested_dictionaries(self, optimal_params: Dict, is_seasonal: bool) -> Tuple[
        Optional[Tuple], Optional[Tuple]]:
        """Рекурсивний пошук параметрів у вкладених словниках."""
        for key, value in optimal_params.items():
            if isinstance(value, dict):
                # Перевіряємо наявність параметрів у вкладеному словнику
                if 'order' in value:
                    order = self._ensure_tuple(value['order'])
                    seasonal_order = None

                    if is_seasonal and 'seasonal_order' in value:
                        seasonal_order = self._ensure_tuple(value['seasonal_order'])
                    elif is_seasonal:
                        seasonal_order = self._build_seasonal_order_from_params(value)

                    self.logger.info(f"Знайдено параметри у вкладеному словнику '{key}'")
                    return order, seasonal_order

                # Перевіряємо окремі параметри у вкладеному словнику
                if all(param_key in value for param_key in ['p', 'd', 'q']):
                    order = (value['p'], value['d'], value['q'])
                    seasonal_order = None

                    if is_seasonal:
                        seasonal_order = (
                            value.get('P', 0),
                            value.get('D', 0),
                            value.get('Q', 0),
                            value.get('s', 7)
                        )

                    self.logger.info(f"Знайдено окремі параметри у вкладеному словнику '{key}'")
                    return order, seasonal_order

        return None, None

    def _build_seasonal_order_from_params(self, params: Dict) -> Optional[Tuple]:
        """Створює seasonal_order з окремих параметрів P, D, Q, s."""
        if any(key in params for key in ['P', 'D', 'Q', 's']):
            return (
                params.get('P', 0),
                params.get('D', 0),
                params.get('Q', 0),
                params.get('s', 7)
            )
        return None

    def _ensure_tuple(self, value) -> Tuple:
        """Перетворює значення в кортеж, якщо це можливо."""
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return value

    def _validate_order_params(self, order: Tuple) -> bool:
        """Валідує параметри order."""
        try:
            if not isinstance(order, (list, tuple)) or len(order) != 3:
                return False

            # Перевіряємо, що всі значення - невід'ємні цілі числа
            p, d, q = order
            if not all(isinstance(x, int) and x >= 0 for x in [p, d, q]):
                return False

            # Перевіряємо розумні межі
            if any(x > 10 for x in [p, d, q]):
                self.logger.warning(f"Параметри order {order} виглядають занадто великими")
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Помилка валідації order: {str(e)}")
            return False

    def _log_structure_for_diagnosis(self, optimal_params: Dict):
        """Детальне логування структури для діагностики."""
        self.logger.debug("Детальна структура optimal_params для діагностики:")
        try:
            for key, value in optimal_params.items():
                if isinstance(value, dict):
                    self.logger.debug(f"  {key} (dict): {list(value.keys())}")
                    # Показуємо вміст вкладених словників
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, dict):
                            self.logger.debug(f"    {nested_key} (dict): {list(nested_value.keys())}")
                        else:
                            self.logger.debug(f"    {nested_key}: {type(nested_value)} = {nested_value}")
                else:
                    self.logger.debug(f"  {key}: {type(value)} = {value}")
        except Exception as log_error:
            self.logger.debug(f"Помилка логування структури: {str(log_error)}")

    def _robust_model_fit(self, model, methods_to_try=['lbfgs', 'bfgs', 'nm']):

        fit_result = None

        for method in methods_to_try:
            self.logger.debug(f"Attempting to fit with method: {method}")

            # Список наборів параметрів для спроб (від найбільш до найменш детального)
            param_sets = [
                {'method': method, 'maxiter': 200, 'disp': False},
                {'method': method, 'maxiter': 200},
                {'method': method, 'disp': False},
                {'method': method},
                {}  # Базовий виклик без параметрів
            ]

            for params in param_sets:
                try:
                    fit_result = model.fit(**params)
                    param_info = f"with parameters: {params}" if params else "with default parameters"
                    self.logger.info(f"Successfully fitted with method: {method} {param_info}")
                    break
                except TypeError as te:
                    if 'unexpected keyword argument' in str(te):
                        continue  # Спробувати наступний набір параметрів
                    else:
                        break  # Інша помилка - перейти до наступного методу
                except Exception as e:
                    self.logger.debug(f"Method {method} with params {params} failed: {str(e)}")
                    break  # Перейти до наступного методу

            if fit_result is not None:
                break

        # Якщо всі методи не спрацювали, спробувати базовий fit
        if fit_result is None:
            self.logger.info("Attempting basic fit without any parameters")
            try:
                fit_result = model.fit()
            except Exception as e:
                self.logger.error(f"Basic fit also failed: {str(e)}")
                raise e

        return fit_result

    def fit_arima(self, data: pd.Series, order: Tuple[int, int, int] = None,
                  symbol: str = 'default', auto_params: bool = True,
                  max_p: int = 5, max_d: int = 2, max_q: int = 5,
                  timeframe: str = "1d") -> Dict:

        # Start with detailed logging of input parameters
        self.logger.info(
            f"Starting ARIMA model fitting for symbol: {symbol}\n"
            f"Parameters - auto_params: {auto_params}, max_p: {max_p}, max_d: {max_d}, max_q: {max_q}, timeframe: {timeframe}\n"
            f"Initial data shape: {data.shape}, first values: {data.head(3).tolist()}, last values: {data.tail(3).tolist()}"
        )

        try:
            # Data validation with logging
            initial_length = len(data)
            data = self._validate_data(data, 10)
            self.logger.debug(
                f"Data validation completed. Initial length: {initial_length}, "
                f"after validation: {len(data)}. Removed {initial_length - len(data)} invalid points"
            )

            # Parameter determination
            if auto_params or order is None:
                self.logger.info("Starting automatic ARIMA parameter determination")

                try:
                    optimal_params = self.ts_analyzer.find_optimal_params(
                        data, max_p=max_p, max_d=max_d, max_q=max_q, seasonal=False
                    )
                    self.logger.debug(f"Optimal params raw result: {optimal_params}")

                    if optimal_params['status'] == 'success':
                        order, _ = self._extract_optimal_params(optimal_params, is_seasonal=False)
                        self.logger.info(
                            f"Successfully determined optimal ARIMA parameters: {order}\n"
                            f"Parameter selection details: {optimal_params.get('details', 'No details available')}"
                        )
                    else:
                        order = (1, 1, 1)
                        self.logger.warning(
                            f"Failed to determine optimal parameters. Reason: {optimal_params.get('message', 'Unknown error')}\n"
                            f"Using default parameters: {order}"
                        )

                except Exception as param_error:
                    order = (1, 1, 1)
                    self.logger.error(
                        f"Error during parameter determination: {str(param_error)}\n"
                        f"Traceback: {traceback.format_exc()}\n"
                        f"Using default parameters: {order}"
                    )

            # Additional data validation with new parameters
            min_required = sum(order) + 10
            pre_validation_length = len(data)
            data = self._validate_data(data, min_required)
            self.logger.debug(
                f"Secondary data validation with order {order} completed. "
                f"Before: {pre_validation_length} points, after: {len(data)} points"
            )

            # Model creation
            model_key = self._generate_model_key("arima", symbol)
            self.logger.info(f"Creating ARIMA model with key: {model_key}")

            try:
                data_values = np.asarray(data, dtype=np.float64)
                self.logger.debug(
                    f"Converted data to numpy array. Shape: {data_values.shape}, dtype: {data_values.dtype}")

                model = StatsARIMA(data_values, order=order)
                self.logger.info(f"ARIMA model initialized with order {order}")

                # Model fitting using robust method
                try:
                    fit_result = self._robust_model_fit(model)

                    # Log convergence information if available
                    if hasattr(fit_result, 'mle_retvals'):
                        convergence_info = fit_result.mle_retvals
                        self.logger.info(f"Convergence info: {convergence_info}")

                except Exception as fitting_error:
                    error_msg = f"Model fitting failed: {str(fitting_error)}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

                # Model evaluation and saving
                params = {"order": order}
                model_info = self._create_model_info(
                    fit_result, data, model_key, "ARIMA", symbol, params, timeframe
                )

                # Log model statistics
                if "stats" in model_info:
                    self.logger.info(
                        "Model statistics:\n"
                        f"AIC: {model_info['stats'].get('aic', 'N/A')}\n"
                        f"BIC: {model_info['stats'].get('bic', 'N/A')}\n"
                        f"HQIC: {model_info['stats'].get('hqic', 'N/A')}\n"
                        f"Log Likelihood: {model_info['stats'].get('llf', 'N/A')}"
                    )

                self.models[model_key] = model_info
                self._save_model_to_db(model_key, model_info)

                self.logger.info(
                    f"ARIMA model {model_key} successfully trained and saved\n"
                    f"Final parameters: {order}\n"
                    f"Training period: {model_info['metadata'].get('start_date', 'N/A')} to "
                    f"{model_info['metadata'].get('end_date', 'N/A')}"
                )

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

            except Exception as model_error:
                self.logger.error(
                    f"Model creation/fitting error: {str(model_error)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise

        except ValueError as ve:
            self.logger.error(
                f"Data validation error: {str(ve)}\n"
                f"Input data stats:\n{data.describe() if isinstance(data, pd.Series) else 'No data available'}"
            )
            return {
                "status": "error",
                "message": str(ve),
                "model_key": None,
                "model_info": None
            }

        except Exception as e:
            self.logger.error(
                f"Unexpected error in ARIMA fitting: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}\n"
                f"Current parameters: order={order}, symbol={symbol}"
            )
            return {
                "status": "error",
                "message": f"Помилка під час навчання моделі ARIMA: {str(e)}",
                "model_key": None,
                "model_info": None
            }

    def fit_sarima(self, data: pd.Series, order: Tuple[int, int, int] = None,
                   seasonal_order: Tuple[int, int, int, int] = None,
                   symbol: str = 'default', auto_params: bool = True,
                   max_p: int = 5, max_d: int = 2, max_q: int = 5,
                   seasonal_period: int = 7, timeframe: str = "1d") -> Dict:

        self.logger.info(
            f"Starting SARIMA model fitting for symbol: {symbol}\n"
            f"Parameters - auto_params: {auto_params}, seasonal_period: {seasonal_period}\n"
            f"Max orders - p: {max_p}, d: {max_d}, q: {max_q}\n"
            f"Initial data shape: {data.shape}, first values: {data.head(3).tolist()}, last values: {data.tail(3).tolist()}"
        )

        try:
            # Data validation
            initial_length = len(data)
            data = self._validate_data(data, 20)
            self.logger.debug(
                f"Data validation completed. Initial length: {initial_length}, "
                f"after validation: {len(data)}. Removed {initial_length - len(data)} invalid points"
            )

            # Parameter determination
            if auto_params or order is None or seasonal_order is None:
                self.logger.info("Starting automatic SARIMA parameter determination")

                try:
                    optimal_params = self.ts_analyzer.find_optimal_params(
                        data, max_p=max_p, max_d=max_d, max_q=max_q, seasonal=True
                    )
                    self.logger.debug(f"Optimal params raw result: {optimal_params}")

                    if optimal_params['status'] == 'success':
                        # ВАЖЛИВО: передаємо is_seasonal=True для SARIMA
                        order, seasonal_order = self._extract_optimal_params(optimal_params, is_seasonal=True)
                        if seasonal_period != 7:
                            seasonal_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period)
                        self.logger.info(
                            f"Successfully determined optimal SARIMA parameters\n"
                            f"Order: {order}, Seasonal order: {seasonal_order}\n"
                            f"Parameter selection details: {optimal_params.get('details', 'No details available')}"
                        )
                    else:
                        order = (1, 1, 1)
                        seasonal_order = (1, 1, 1, seasonal_period)
                        self.logger.warning(
                            f"Failed to determine optimal parameters. Reason: {optimal_params.get('message', 'Unknown error')}\n"
                            f"Using default parameters: order={order}, seasonal_order={seasonal_order}"
                        )

                except Exception as param_error:
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, seasonal_period)
                    self.logger.error(
                        f"Error during parameter determination: {str(param_error)}\n"
                        f"Traceback: {traceback.format_exc()}\n"
                        f"Using default parameters: order={order}, seasonal_order={seasonal_order}"
                    )

            # Additional data validation
            min_required = sum(order) + sum(seasonal_order[:-1]) + 2 * seasonal_order[-1]
            pre_validation_length = len(data)
            data = self._validate_data(data, min_required)
            self.logger.debug(
                f"Secondary data validation completed. Required: {min_required} points\n"
                f"Before: {pre_validation_length} points, after: {len(data)} points"
            )

            # Model creation
            model_key = self._generate_model_key("sarima", symbol)
            self.logger.info(f"Creating SARIMA model with key: {model_key}")

            try:
                data_values = np.asarray(data, dtype=np.float64)
                self.logger.debug(
                    f"Converted data to numpy array. Shape: {data_values.shape}, dtype: {data_values.dtype}")

                model = SARIMAX(
                    data_values,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.logger.info(
                    f"SARIMA model initialized with order {order} and seasonal order {seasonal_order}"
                )

                # Model fitting using robust method with SARIMA-specific parameters
                try:
                    # Для SARIMA використовуємо розширений набір параметрів
                    def _robust_sarima_fit(model, methods_to_try=['lbfgs', 'bfgs', 'nm']):
                        fit_result = None

                        for method in methods_to_try:
                            self.logger.debug(f"Attempting to fit SARIMA with method: {method}")

                            # Набори параметрів специфічні для SARIMAX
                            param_sets = [
                                {'method': method, 'maxiter': 200, 'disp': False, 'warn_convergence': False},
                                {'method': method, 'maxiter': 200, 'disp': False},
                                {'method': method, 'maxiter': 200},
                                {'method': method, 'disp': False},
                                {'method': method},
                                {}  # Базовий виклик
                            ]

                            for params in param_sets:
                                try:
                                    fit_result = model.fit(**params)
                                    param_info = f"with parameters: {params}" if params else "with default parameters"
                                    self.logger.info(f"Successfully fitted SARIMA with method: {method} {param_info}")
                                    break
                                except TypeError as te:
                                    if 'unexpected keyword argument' in str(te):
                                        continue
                                    else:
                                        break
                                except Exception:
                                    break

                            if fit_result is not None:
                                break

                        if fit_result is None:
                            self.logger.info("Attempting basic SARIMA fit without any parameters")
                            fit_result = model.fit()

                        return fit_result

                    fit_result = _robust_sarima_fit(model)

                    # Log convergence information if available
                    if hasattr(fit_result, 'mle_retvals'):
                        convergence_info = fit_result.mle_retvals
                        self.logger.info(f"Convergence info: {convergence_info}")

                except Exception as fitting_error:
                    error_msg = f"SARIMA model fitting failed: {str(fitting_error)}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

                # Model evaluation and saving
                params = {
                    "order": order,
                    "seasonal_order": seasonal_order
                }
                model_info = self._create_model_info(
                    fit_result, data, model_key, "SARIMA", symbol, params, timeframe
                )

                # Log model statistics
                if "stats" in model_info:
                    self.logger.info(
                        "Model statistics:\n"
                        f"AIC: {model_info['stats'].get('aic', 'N/A')}\n"
                        f"BIC: {model_info['stats'].get('bic', 'N/A')}\n"
                        f"HQIC: {model_info['stats'].get('hqic', 'N/A')}\n"
                        f"Log Likelihood: {model_info['stats'].get('llf', 'N/A')}"
                    )

                self.models[model_key] = model_info
                self._save_model_to_db(model_key, model_info)

                self.logger.info(
                    f"SARIMA model {model_key} successfully trained and saved\n"
                    f"Final parameters: order={order}, seasonal_order={seasonal_order}\n"
                    f"Training period: {model_info['metadata'].get('start_date', 'N/A')} to "
                    f"{model_info['metadata'].get('end_date', 'N/A')}"
                )

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

            except Exception as model_error:
                self.logger.error(
                    f"Model creation/fitting error: {str(model_error)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise

        except ValueError as ve:
            self.logger.error(
                f"Data validation error: {str(ve)}\n"
                f"Input data stats:\n{data.describe() if isinstance(data, pd.Series) else 'No data available'}"
            )
            return {
                "status": "error",
                "message": str(ve),
                "model_key": None,
                "model_info": None
            }

        except Exception as e:
            self.logger.error(
                f"Unexpected error in SARIMA fitting: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}\n"
                f"Current parameters: order={order}, seasonal_order={seasonal_order}, symbol={symbol}"
            )
            return {
                "status": "error",
                "message": f"Помилка під час навчання моделі SARIMA: {str(e)}",
                "model_key": None,
                "model_info": None
            }

    def get_model_forecast(self, model_key: str, steps: int,
                           return_conf_int: bool = True, alpha: float = 0.05) -> Dict:

        self.logger.info(
            f"Getting forecast for model {model_key}\n"
            f"Parameters - steps: {steps}, return_conf_int: {return_conf_int}, alpha: {alpha}"
        )

        # Check model availability
        if model_key not in self.models:
            if self.db_manager is not None:
                try:
                    self.logger.info(f"Model {model_key} not in memory, attempting DB load")
                    loaded = self.db_manager.load_complete_model(model_key)

                    if loaded:
                        self.logger.info(f"Successfully loaded model {model_key} from DB")
                    else:
                        error_msg = f"Failed to load model {model_key} from database"
                        self.logger.error(error_msg)
                        return {"status": "error", "message": error_msg}

                except Exception as e:
                    error_msg = f"Database error loading model {model_key}: {str(e)}"
                    self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
                    return {"status": "error", "message": error_msg}
            else:
                error_msg = f"Model {model_key} not found and no DB manager available"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

        try:
            model_info = self.models[model_key]
            fit_result = model_info["fit_result"]

            self.logger.debug(
                f"Model info - type: {model_info['metadata'].get('model_type')}, "
                f"timeframe: {model_info['metadata'].get('timeframe')}, "
                f"params: {model_info['parameters']}"
            )

            # Get forecast
            if return_conf_int:
                self.logger.debug("Generating forecast with confidence intervals")
                forecast = fit_result.forecast(steps=steps)
                conf_int = fit_result.get_forecast(steps=steps).conf_int(alpha=alpha)

                self.logger.info(
                    f"Successfully generated forecast for {steps} steps\n"
                    f"First 3 forecast values: {forecast[:3] if hasattr(forecast, '__getitem__') else 'N/A'}\n"
                    f"Last 3 forecast values: {forecast[-3:] if hasattr(forecast, '__getitem__') else 'N/A'}"
                )

                result = {
                    "status": "success",
                    "forecast": forecast.tolist() if isinstance(forecast, np.ndarray) else forecast,
                    "conf_int_lower": conf_int.iloc[:, 0].tolist(),
                    "conf_int_upper": conf_int.iloc[:, 1].tolist(),
                    "alpha": alpha
                }
            else:
                self.logger.debug("Generating forecast without confidence intervals")
                forecast = fit_result.forecast(steps=steps)

                self.logger.info(
                    f"Successfully generated forecast for {steps} steps\n"
                    f"First 3 forecast values: {forecast[:3] if hasattr(forecast, '__getitem__') else 'N/A'}\n"
                    f"Last 3 forecast values: {forecast[-3:] if hasattr(forecast, '__getitem__') else 'N/A'}"
                )

                result = {
                    "status": "success",
                    "forecast": forecast.tolist() if isinstance(forecast, np.ndarray) else forecast
                }

            return result

        except Exception as e:
            error_msg = f"Error generating forecast: {str(e)}"
            self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            return {"status": "error", "message": error_msg}

    def save_model(self, model_key: str, path: str) -> bool:

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
                    # Створюємо бінарний представлення моделі
                    model_binary = pickle.dumps(loaded_data["fit_result"])

                    # Отримуємо метадані з моделі або створюємо базові
                    metadata = loaded_data.get("metadata", {})

                    # Забезпечуємо наявність всіх обов'язкових полів
                    required_metadata = {
                        'model_type': metadata.get('model_type', 'ARIMA'),
                        'timeframe': metadata.get('timeframe', '1D'),
                        'start_date': metadata.get('start_date', '2020-01-01'),
                        'end_date': metadata.get('end_date', '2024-01-01')
                    }

                    # Оновлюємо метадані
                    metadata.update(required_metadata)

                    # Зберігаємо всі компоненти моделі з правильними аргументами
                    self.db_manager.save_model_metadata(
                        model_key=model_key,
                        model_type=metadata['model_type'],
                        timeframe=metadata['timeframe'],
                        start_date=metadata['start_date'],
                        end_date=metadata['end_date'],
                        **{k: v for k, v in metadata.items() if
                           k not in ['model_type', 'timeframe', 'start_date', 'end_date']}
                    )
                    self.db_manager.save_model_parameters(model_key, loaded_data["parameters"])
                    self.db_manager.save_model_binary(model_key, model_binary)

                    # Зберігаємо метрики, якщо є
                    if "stats" in loaded_data and loaded_data["stats"]:
                        self.db_manager.save_model_metrics(model_key, loaded_data["stats"])

                    # Зберігаємо трансформації, якщо є
                    if "transformations" in loaded_data and loaded_data["transformations"]:
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

    def select_best_stationary_column(self, data: pd.DataFrame, symbol: str):

        # Спочатку перевіряємо стовпці, які вже відзначені як стаціонарні
        if 'is_stationary' in data.columns:
            stationary_mask = data['is_stationary'] == True
            if stationary_mask.any():
                stationary_indices = data.index[stationary_mask]
                # Знаходимо стовпець з найкращими AIC/BIC, якщо вони доступні
                if 'aic_score' in data.columns and 'bic_score' in data.columns:
                    stationary_data = data.loc[stationary_mask]
                    best_idx = stationary_data['aic_score'].idxmin()
                    # Повертаємо назву стовпця-кандидата, а не значення
                    return data.index[best_idx] if best_idx in data.index else stationary_indices[0]
                return stationary_indices[0]

        # Якщо немає колонки is_stationary, перевіряємо p-значення тестів
        if 'adf_pvalue' in data.columns:
            stationary_mask = data['adf_pvalue'] < 0.05
            if stationary_mask.any():
                return data.index[stationary_mask][0]

        # Якщо аналіз не дав результатів, використовуємо стандартні перетворення в порядку їх ефективності
        for col in ['close_diff2', 'close_seasonal_diff', 'close_diff', 'close_log_diff', 'close_combo_diff',
                    'close_log', 'original_close']:
            if col in data.index:  # використовуємо index замість columns
                return col

        # Якщо жоден з бажаних стовпців не знайдено
        raise ValueError("Не знайдено відповідного стовпця для ARIMA моделювання")

    def auto_determine_order(self, data: pd.Series, max_order: int = 5):

        try:
            # Переконуємося, що дані є числовими
            if not pd.api.types.is_numeric_dtype(data):
                # Спробуємо конвертувати в числовий тип
                data = pd.to_numeric(data, errors='coerce')
                # Видаляємо NaN значення
                data = data.dropna()

            # Перевіряємо, чи залишилися дані після очищення
            if len(data) == 0:
                self.logger.error("Немає валідних числових даних для аналізу")
                return (1, 1, 1)

            # Переконуємося, що індекс є числовим або datetime
            if not isinstance(data.index, (pd.DatetimeIndex, pd.RangeIndex)):
                data.index = pd.RangeIndex(len(data))

            # FIXED: Використовуємо pmdarima.auto_arima для визначення оптимальних параметрів
            from pmdarima import auto_arima

            # Використовуємо auto_arima для визначення оптимальних параметрів
            automodel = auto_arima(
                data,
                start_p=0, start_q=0,
                max_p=max_order, max_q=max_order,
                d=None,  # auto-detection
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            return automodel.order
        except Exception as e:
            self.logger.error(f"Помилка під час автоматичного визначення порядку: {str(e)}")
            # Повертаємо стандартні параметри у випадку помилки
            return (1, 1, 1)

    def apply_transformations(self, data: pd.Series, model_key: str) -> pd.Series:

        if model_key not in self.transformations:
            return data

        # Переконуємося, що дані є числовими
        if not pd.api.types.is_numeric_dtype(data):
            data = pd.to_numeric(data, errors='coerce')
            data = data.dropna()

        transformed_data = data.copy()
        for transform_id, transform_info in self.transformations[model_key].items():
            transform_type = transform_info.get('type')

            try:
                if transform_type == 'diff':
                    order = transform_info.get('order', 1)
                    transformed_data = transformed_data.diff(order).dropna()
                elif transform_type == 'log':
                    # Переконуємося, що всі значення додатні для логарифма
                    if (transformed_data <= 0).any():
                        self.logger.warning("Виявлено від'ємні або нульові значення для логарифмічної трансформації")
                        transformed_data = transformed_data[transformed_data > 0]
                    transformed_data = np.log(transformed_data)
                elif transform_type == 'seasonal_diff':
                    period = transform_info.get('period', 1)
                    transformed_data = transformed_data.diff(period).dropna()
            except Exception as e:
                self.logger.error(f"Помилка застосування трансформації {transform_type}: {str(e)}")
                continue

        return transformed_data

    def inverse_transformations(self, forecasted_data: pd.Series, model_key: str,
                                original_data: pd.Series) -> pd.Series:

        if model_key not in self.transformations:
            return forecasted_data

        # Переконуємося, що дані є числовими
        if not pd.api.types.is_numeric_dtype(original_data):
            original_data = pd.to_numeric(original_data, errors='coerce')
            original_data = original_data.dropna()

        # Трансформації потрібно скасовувати в зворотному порядку
        inverse_transforms = list(self.transformations[model_key].items())
        inverse_transforms.reverse()

        result = forecasted_data.copy()

        for transform_id, transform_info in inverse_transforms:
            transform_type = transform_info.get('type')

            try:
                if transform_type == 'diff':
                    order = transform_info.get('order', 1)
                    # Для інверсії диференціювання потрібне початкове значення
                    if len(original_data) >= order:
                        last_values = original_data.iloc[-order:].values
                        for i in range(len(result)):
                            result.iloc[i] = result.iloc[i] + last_values[i % order]
                elif transform_type == 'log':
                    result = np.exp(result)
                elif transform_type == 'seasonal_diff':
                    period = transform_info.get('period', 1)
                    # Інверсія сезонного диференціювання потребує сезонних значень
                    if len(original_data) >= period:
                        seasonal_values = original_data.iloc[-period:].values
                        for i in range(len(result)):
                            result.iloc[i] = result.iloc[i] + seasonal_values[i % period]
            except Exception as e:
                self.logger.error(f"Помилка інверсії трансформації {transform_type}: {str(e)}")
                continue

        return result