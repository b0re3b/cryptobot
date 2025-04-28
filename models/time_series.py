import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TimeSeriesModels:
    """
    Клас для моделювання часових рядів криптовалют з використанням класичних методів.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Ініціалізація класу моделей часових рядів.

        Args:
            log_level: Рівень логування
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.models = {}

    def check_stationarity(self, data: pd.Series) -> Dict:
        """
        Перевірка стаціонарності часового ряду.

        Args:
            data: Часовий ряд (pandas Series)

        Returns:
            Словник з результатами тестів на стаціонарність
        """
        from statsmodels.tsa.stattools import adfuller, kpss

        adf_results = adfuller(data)
        kpss_results = kpss(data)

        return {
            'adf_statistic': adf_results[0],
            'adf_pvalue': adf_results[1],
            'adf_is_stationary': adf_results[1] < 0.05,
            'kpss_statistic': kpss_results[0],
            'kpss_pvalue': kpss_results[1],
            'kpss_is_stationary': kpss_results[1] > 0.05
        }

    def difference_series(self, data: pd.Series, order: int = 1) -> pd.Series:
        """
        Диференціювання часового ряду для досягнення стаціонарності.

        Args:
            data: Часовий ряд
            order: Порядок диференціювання

        Returns:
            Диференційований часовий ряд
        """
        return data.diff(order).dropna()

    def find_optimal_params(self, data: pd.Series, max_p: int = 5, max_d: int = 2,
                            max_q: int = 5, seasonal: bool = False) -> Dict:
        """
        Пошук оптимальних параметрів для ARIMA/SARIMA.

        Args:
            data: Часовий ряд
            max_p, max_d, max_q: Максимальні значення параметрів ARIMA
            seasonal: Чи враховувати сезонність

        Returns:
            Словник з оптимальними параметрами
        """
        model = auto_arima(
            data,
            start_p=0, start_q=0,
            max_p=max_p, max_d=max_d, max_q=max_q,
            seasonal=seasonal,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        return {
            'order': model.order,
            'seasonal_order': model.seasonal_order if seasonal else None,
            'aic': model.aic(),
            'bic': model.bic()
        }

    def fit_arima(self, data: pd.Series, order: Tuple[int, int, int],
                  symbol: str = 'default') -> Dict:
        """
        Навчання моделі ARIMA.

        Args:
            data: Часовий ряд
            order: Параметри ARIMA (p, d, q)
            symbol: Ідентифікатор моделі

        Returns:
            Результати навчання
        """
        model = ARIMA(data, order=order)
        fitted_model = model.fit()

        self.models[f'arima_{symbol}'] = fitted_model

        return {
            'model': fitted_model,
            'params': order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }

    def fit_sarima(self, data: pd.Series, order: Tuple[int, int, int],
                   seasonal_order: Tuple[int, int, int, int], symbol: str = 'default') -> Dict:
        """
        Навчання моделі SARIMA.

        Args:
            data: Часовий ряд
            order: Параметри ARIMA (p, d, q)
            seasonal_order: Сезонні параметри (P, D, Q, s)
            symbol: Ідентифікатор моделі

        Returns:
            Результати навчання
        """
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)

        self.models[f'sarima_{symbol}'] = fitted_model

        return {
            'model': fitted_model,
            'params': {
                'order': order,
                'seasonal_order': seasonal_order
            },
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }

    def forecast(self, model_key: str, steps: int = 24) -> pd.Series:
        """
        Прогнозування на основі навченої моделі.

        Args:
            model_key: Ключ моделі в self.models
            steps: Кількість кроків для прогнозу

        Returns:
            Прогнозні значення
        """
        if model_key not in self.models:
            self.logger.error(f"Модель {model_key} не знайдена")
            return None

        forecast = self.models[model_key].forecast(steps=steps)
        return forecast

    def evaluate_model(self, model_key: str, test_data: pd.Series) -> Dict:
        """
        Оцінка точності моделі.

        Args:
            model_key: Ключ моделі в self.models
            test_data: Тестові дані для оцінки

        Returns:
            Метрики точності
        """
        if model_key not in self.models:
            self.logger.error(f"Модель {model_key} не знайдена")
            return None

        forecast = self.forecast(model_key, steps=len(test_data))

        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, forecast)
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def save_model(self, model_key: str, path: str) -> bool:
        """
        Збереження моделі на диск.

        Args:
            model_key: Ключ моделі в self.models
            path: Шлях для збереження

        Returns:
            Успішність операції
        """
        if model_key not in self.models:
            self.logger.error(f"Модель {model_key} не знайдена")
            return False

        import joblib
        try:
            joblib.dump(self.models[model_key], path)
            return True
        except Exception as e:
            self.logger.error(f"Помилка збереження моделі: {e}")
            return False

    def load_model(self, model_key: str, path: str) -> bool:
        """
        Завантаження моделі з диску.

        Args:
            model_key: Ключ для збереження моделі
            path: Шлях до файлу моделі

        Returns:
            Успішність операції
        """
        import joblib
        try:
            model = joblib.load(path)
            self.models[model_key] = model
            return True
        except Exception as e:
            self.logger.error(f"Помилка завантаження моделі: {e}")
            return False