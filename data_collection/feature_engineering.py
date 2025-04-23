import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Callable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import logging
import ta
from data.db import DatabaseManager
from utils.config import db_connection

class FeatureEngineering:

    def __init__(self, log_level=logging.INFO):
        self.log_level = log_level
        self.db_connection = db_connection
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols
        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")
        self.ready = True

    def create_lagged_features(self, data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               lag_periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        self.logger.info("Створення лагових ознак...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що індекс часовий для правильного зсуву
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є часовим (DatetimeIndex). Лагові ознаки можуть бути неточними.")

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних
            missing_cols = [col for col in columns if col not in result_df.columns]
            if missing_cols:
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
                columns = [col for col in columns if col in result_df.columns]

        # Створюємо лагові ознаки для кожного стовпця і періоду
        for col in columns:
            for lag in lag_periods:
                lag_col_name = f"{col}_lag_{lag}"
                result_df[lag_col_name] = result_df[col].shift(lag)
                self.logger.debug(f"Створено лаг {lag} для стовпця {col}")

        # Інформуємо про кількість доданих ознак
        num_added_features = len(columns) * len(lag_periods)
        self.logger.info(f"Додано {num_added_features} лагових ознак")

        return result_df

    def create_rolling_features(self, data: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                window_sizes: List[int] = [5, 10, 20, 50],
                                functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:

        self.logger.info("Створення ознак на основі ковзного вікна...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що індекс часовий для правильного розрахунку
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning(
                "Індекс даних не є часовим (DatetimeIndex). Ознаки ковзного вікна можуть бути неточними.")

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних
            missing_cols = [col for col in columns if col not in result_df.columns]
            if missing_cols:
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
                columns = [col for col in columns if col in result_df.columns]

        # Словник функцій pandas для ковзного вікна
        func_map = {
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            'max': 'max',
            'median': 'median',
            'sum': 'sum',
            'var': 'var',
            'kurt': 'kurt',
            'skew': 'skew',
            'quantile_25': lambda x: x.quantile(0.25),
            'quantile_75': lambda x: x.quantile(0.75)
        }

        # Перевіряємо, чи всі функції підтримуються
        unsupported_funcs = [f for f in functions if f not in func_map]
        if unsupported_funcs:
            self.logger.warning(f"Функції {unsupported_funcs} не підтримуються і будуть пропущені")
            functions = [f for f in functions if f in func_map]

        # Лічильник доданих ознак
        added_features_count = 0

        # Для кожного стовпця, розміру вікна і функції створюємо нову ознаку
        for col in columns:
            for window in window_sizes:
                # Створюємо об'єкт ковзного вікна
                rolling_window = result_df[col].rolling(window=window, min_periods=1)

                for func_name in functions:
                    # Отримуємо функцію з мапінгу
                    func = func_map[func_name]

                    # Створюємо нову ознаку
                    feature_name = f"{col}_rolling_{window}_{func_name}"

                    # Застосовуємо функцію до ковзного вікна
                    if callable(func):
                        result_df[feature_name] = rolling_window.apply(func)
                    else:
                        result_df[feature_name] = getattr(rolling_window, func)()

                    added_features_count += 1
                    self.logger.debug(f"Створено ознаку {feature_name}")

        # Обробляємо NaN значення на початку часового ряду
        # Заповнюємо перші значення медіаною стовпця (можна змінити на інший метод)
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                result_df[col] = result_df[col].fillna(result_df[col].median())

        self.logger.info(f"Додано {added_features_count} ознак ковзного вікна")

        return result_df

    def create_ewm_features(self, data: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            spans: List[int] = [5, 10, 20, 50],
                            functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Створює ознаки на основі експоненціально зваженого вікна.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        columns : list of str, optional
            Список стовпців для створення ознак (якщо None, використовуються всі числові)
        spans : list of int
            Значення span для EWM
        functions : list of str
            Функції для обчислення ('mean', 'std', тощо)

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими EWM ознаками
        """
        # Вибрати числові стовпці, якщо columns не вказано
        # Для кожного стовпця, span і функції створити нову ознаку
        # Перевірити наявність NaN значень і обробити їх
        pass

    def create_return_features(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        """
        Створює ознаки прибутковості за різні періоди.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        price_column : str
            Стовпець з ціною для розрахунку прибутковості
        periods : list of int
            Список періодів для розрахунку прибутковості

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими ознаками прибутковості
        """
        # Перевірити, що price_column існує в даних
        # Розрахувати процентну зміну для кожного періоду
        # Опціонально додати логарифмічну прибутковість
        pass

    def create_technical_features(self, data: pd.DataFrame,
                                  indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Створює ознаки на основі технічних індикаторів.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame з OHLCV даними
        indicators : list of str, optional
            Список індикаторів для розрахунку (якщо None, використовується базовий набір)

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими технічними індикаторами
        """
        # Перевірити, що необхідні OHLCV колонки існують
        # Розрахувати вибрані технічні індикатори
        # Використати talib або власні реалізації
        pass

    def create_volatility_features(self, data: pd.DataFrame,
                                   price_column: str = 'close',
                                   window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Створює ознаки волатильності.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        price_column : str
            Стовпець з ціною для розрахунку волатильності
        window_sizes : list of int
            Розміри вікон для обчислення волатильності

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими ознаками волатильності
        """
        # Розрахувати волатильність як стандартне відхилення прибутковості
        # Створити ознаки для різних розмірів вікон
        # Додати інші метрики волатильності (наприклад, Garman-Klass)
        pass

    def create_ratio_features(self, data: pd.DataFrame,
                              numerators: List[str],
                              denominators: List[str]) -> pd.DataFrame:
        """
        Створює ознаки на основі співвідношень між різними метриками.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        numerators : list of str
            Список стовпців для чисельника
        denominators : list of str
            Список стовпців для знаменника

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими ознаками-співвідношеннями
        """
        # Перевірити, що всі зазначені стовпці існують
        # Створити всі можливі комбінації співвідношень
        # Обробити випадки з нульовими знаменниками
        pass

    def create_crossover_features(self, data: pd.DataFrame,
                                  fast_columns: List[str],
                                  slow_columns: List[str]) -> pd.DataFrame:
        """
        Створює ознаки на основі перетинів індикаторів.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        fast_columns : list of str
            Список "швидких" індикаторів
        slow_columns : list of str
            Список "повільних" індикаторів

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими ознаками перетинів
        """
        # Перевірити, що всі зазначені стовпці існують
        # Створити ознаки, що вказують на перетин (наприклад, 1 при golden cross)
        # Додати ознаки відстані між індикаторами
        pass

    def create_candle_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Створює ознаки на основі патернів свічок.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame з OHLCV даними

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими ознаками патернів свічок
        """
        # Перевірити, що необхідні OHLCV колонки існують
        # Використати talib для визначення патернів свічок
        # Опціонально створити власні патерни
        pass

    def create_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Створює специфічні для криптовалют індикатори.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими специфічними індикаторами
        """
        # Розрахувати CMF (Chaikin Money Flow)
        # Розрахувати OBV (On Balance Volume)
        # Інші специфічні для крипто індикатори
        pass

    def create_volume_features(self, data: pd.DataFrame,
                               window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Створює ознаки на основі об'єму.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame з колонкою volume
        window_sizes : list of int
            Розміри вікон для обчислення

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими ознаками об'єму
        """
        # Перевірити, що колонка volume існує
        # Створити ознаки відносного об'єму
        # Створити ознаки ковзного середнього об'єму
        pass

    def create_datetime_features(self, data: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
        """
        Створює ознаки на основі дати і часу.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame з часовим індексом
        cyclical : bool
            Чи створювати циклічні ознаки через sin/cos трансформації

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими часовими ознаками
        """
        # Перевірити, що індекс є DatetimeIndex
        # Створити базові ознаки (година, день тижня, місяць)
        # Опціонально створити циклічні ознаки
        pass

    def create_target_variable(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               horizon: int = 1,
                               target_type: str = 'return') -> pd.DataFrame:
        """
        Створює цільову змінну для прогнозування.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        price_column : str
            Стовпець з ціною для створення цілі
        horizon : int
            Горизонт прогнозування (кількість періодів вперед)
        target_type : str
            Тип цільової змінної ('return', 'direction', 'volatility')

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданою цільовою змінною
        """
        # Створити цільову змінну в залежності від target_type
        # 'return': процентна зміна ціни
        # 'direction': 1 якщо ціна зросла, 0 якщо знизилась
        # 'volatility': майбутня волатильність
        pass

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        n_features: Optional[int] = None,
                        method: str = 'f_regression') -> Tuple[pd.DataFrame, List[str]]:
        """
        Вибирає найбільш важливі ознаки для моделювання.

        Parameters:
        -----------
        X : pandas.DataFrame
            DataFrame з ознаками
        y : pandas.Series
            Цільова змінна
        n_features : int, optional
            Кількість ознак для вибору (якщо None, використовується половина)
        method : str
            Метод вибору ознак ('f_regression', 'mutual_info', 'rfe')

        Returns:
        --------
        tuple
            (DataFrame з відібраними ознаками, список назв вибраних ознак)
        """
        # Визначити кількість ознак для вибору, якщо не вказано
        # Застосувати вибраний метод відбору ознак
        # Повернути відібрані ознаки і їх назви
        pass

    def reduce_dimensions(self, data: pd.DataFrame,
                          n_components: Optional[int] = None,
                          method: str = 'pca') -> Tuple[pd.DataFrame, object]:
        """
        Зменшує розмірність набору ознак.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame з ознаками
        n_components : int, optional
            Кількість компонентів (якщо None, вибирається автоматично)
        method : str
            Метод зменшення розмірності ('pca', 't-sne', 'umap')

        Returns:
        --------
        tuple
            (DataFrame зі зменшеною розмірністю, об'єкт трансформатора)
        """
        # Вибрати і застосувати метод зменшення розмірності
        # Створити новий DataFrame з новими компонентами
        # Повернути трансформовані дані і трансформатор
        pass

    def create_polynomial_features(self, data: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   degree: int = 2,
                                   interaction_only: bool = False) -> pd.DataFrame:
        """
        Створює поліноміальні ознаки та взаємодії.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        columns : list of str, optional
            Список стовпців для створення ознак (якщо None, використовуються всі числові)
        degree : int
            Ступінь поліному
        interaction_only : bool
            Чи створювати тільки взаємодії без степенів

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими поліноміальними ознаками
        """
        # Вибрати числові стовпці, якщо columns не вказано
        # Створити і застосувати PolynomialFeatures
        # Повернути DataFrame з новими ознаками
        pass

    def create_cluster_features(self, data: pd.DataFrame,
                                n_clusters: int = 5,
                                method: str = 'kmeans') -> pd.DataFrame:
        """
        Створює ознаки на основі кластеризації.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame
        n_clusters : int
            Кількість кластерів
        method : str
            Метод кластеризації ('kmeans', 'dbscan', 'hierarchical')

        Returns:
        --------
        pandas.DataFrame
            DataFrame з доданими кластерними ознаками
        """
        # Вибрати метод кластеризації і застосувати його
        # Додати мітки кластерів як нові ознаки
        # Додати відстань до центроїдів кластерів як ознаки
        pass

    def prepare_features_pipeline(self, data: pd.DataFrame,
                                  target_column: str = 'close',
                                  horizon: int = 1,
                                  feature_groups: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Створює повний набір ознак за допомогою конвеєра обробки.

        Parameters:
        -----------
        data : pandas.DataFrame
            Вхідний DataFrame з OHLCV даними
        target_column : str
            Назва стовпця для створення цільової змінної
        horizon : int
            Горизонт прогнозування
        feature_groups : list of str, optional
            Список груп ознак для включення

        Returns:
        --------
        tuple
            (DataFrame з ознаками, Series з цільовою змінною)
        """
        # Визначити стандартні групи ознак, якщо не вказано
        # Створити ознаки з кожної групи
        # Створити цільову змінну
        # Повернути набір ознак і цільову змінну
        pass

def main():