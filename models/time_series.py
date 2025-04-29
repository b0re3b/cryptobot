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
    - get_orderbook_processed: для отримання обробленої книги ордерів
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
        """
        Пошук оптимальних параметрів для ARIMA/SARIMA.

        Args:
            data: Часовий ряд
            max_p, max_d, max_q: Максимальні значення параметрів ARIMA
            seasonal: Чи враховувати сезонність

        Returns:
            Словник з оптимальними параметрами
        """
        pass

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

        Примітка:
        - Після навчання моделі рекомендується зберегти її в БД за допомогою self.db_manager.save_model_metadata,
          self.db_manager.save_model_parameters та self.db_manager.save_model_binary
        - Ключ моделі для збереження повинен бути у форматі f"{symbol}_arima_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        """
        pass

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

        Примітка:
        - Після навчання моделі рекомендується зберегти її в БД за допомогою self.db_manager.save_model_metadata,
          self.db_manager.save_model_parameters та self.db_manager.save_model_binary
        - Ключ моделі для збереження повинен бути у форматі f"{symbol}_sarima_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        """
        pass

    def forecast(self, model_key: str, steps: int = 24) -> pd.Series:
        """
        Прогнозування на основі навченої моделі.

        Args:
            model_key: Ключ моделі в self.models
            steps: Кількість кроків для прогнозу

        Returns:
            Прогнозні значення

        Примітка:
        - Спочатку потрібно завантажити модель з БД за допомогою self.db_manager.load_complete_model(model_key)
        - Після прогнозування рекомендується зберегти результати в БД за допомогою self.db_manager.save_model_forecasts
        """
        pass

    def evaluate_model(self, model_key: str, test_data: pd.Series) -> Dict:
        """
        Оцінка точності моделі.

        Args:
            model_key: Ключ моделі в self.models
            test_data: Тестові дані для оцінки

        Returns:
            Метрики точності

        Примітка:
        - Спочатку потрібно завантажити модель з БД за допомогою self.db_manager.load_complete_model(model_key)
        - Після оцінки рекомендується зберегти метрики в БД за допомогою self.db_manager.save_model_metrics
        - Метрики мають включати MSE, RMSE, MAE, MAPE
        """
        pass

    def save_model(self, model_key: str, path: str) -> bool:
        """
        Збереження моделі на диск.

        Args:
            model_key: Ключ моделі в self.models
            path: Шлях для збереження

        Returns:
            Успішність операції

        Примітка:
        - Спочатку потрібно завантажити модель з БД за допомогою self.db_manager.load_complete_model(model_key)
        - Після завантаження модель зберігається на диск
        """
        pass

    def load_model(self, model_key: str, path: str) -> bool:
        """
        Завантаження моделі з диску.

        Args:
            model_key: Ключ для збереження моделі
            path: Шлях до файлу моделі

        Returns:
            Успішність операції

        Примітка:
        - Після завантаження з диску модель має бути збережена в БД за допомогою self.db_manager.save_complete_model
        """
        pass

    def transform_data(self, data: pd.Series, method: str = 'log') -> Union[pd.Series, Tuple[pd.Series, float]]:
        """
        Трансформація даних для стабілізації дисперсії.

        Args:
            data: Часовий ряд
            method: Метод трансформації ('log', 'sqrt', 'boxcox')

        Returns:
            Трансформований часовий ряд або кортеж (трансформований ряд, параметр трансформації)

        Примітка:
        - При використанні у pipeline навчання моделі, слід зберегти інформацію про трансформацію
          за допомогою self.db_manager.save_data_transformations для можливості зворотної трансформації прогнозів
        """
        pass

    def inverse_transform(self, data: pd.Series, method: str = 'log', lambda_param: float = None) -> pd.Series:
        """
        Зворотна трансформація даних.

        Args:
            data: Трансформований часовий ряд
            method: Метод зворотної трансформації
            lambda_param: Параметр для BoxCox

        Returns:
            Зворотно трансформований часовий ряд
        """
        pass

    def detect_seasonality(self, data: pd.Series) -> Dict:
        """
        Виявлення сезонності в часовому ряді.

        Args:
            data: Часовий ряд

        Returns:
            Словник з результатами аналізу сезонності
        """
        pass

    def rolling_window_validation(self, data: pd.Series, model_type: str = 'arima',
                                  order: Tuple = None, seasonal_order: Tuple = None,
                                  window_size: int = 100, step: int = 20,
                                  forecast_horizon: int = 10) -> Dict:
        """
        Ковзаюча валідація моделі для часових рядів.

        Args:
            data: Часовий ряд
            model_type: Тип моделі ('arima', 'sarima')
            order: Параметри ARIMA
            seasonal_order: Сезонні параметри
            window_size: Розмір вікна для навчання
            step: Крок для зсуву вікна
            forecast_horizon: Горизонт прогнозу

        Returns:
            Результати валідації

        Примітка:
        - Результати валідації можуть бути збережені в БД за допомогою self.db_manager.save_model_metrics
          для кожної ітерації з різними датами тестування
        """
        pass

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