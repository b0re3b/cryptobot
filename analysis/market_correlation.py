from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from DMP.market_data_processor import MarketDataProcessor
from data.db import DatabaseManager
from utils.config import *
from utils.logger import CryptoLogger


class MarketCorrelation:

    def __init__(self):
        self.logger = CryptoLogger('correlation')
        self.db_manager = DatabaseManager()
        self.data_processor = MarketDataProcessor()

        self.logger.info("Ініціалізація аналізатора ринкової кореляції")

        self.config = DEFAULT_CONFIG.copy()


    def _update_config_recursive(self, target_dict: Dict, source_dict: Dict) -> None:
        """
           Рекурсивно оновлює словник конфігурації новими значеннями з іншого словника.

           Якщо ключ присутній у обох словниках і обидва значення є словниками,
           відбувається рекурсивне оновлення. Інакше значення з source_dict переписує
           значення в target_dict.

           Parameters:
               target_dict (Dict): Цільовий словник, який буде оновлено.
               source_dict (Dict): Словник із новими значеннями.

           Returns:
               None
           """
        for key, value in source_dict.items():
            if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target_dict[key], value)
            else:
                self.logger.debug(f"Оновлення параметра конфігурації: {key} = {value}")
                target_dict[key] = value

    def calculate_price_correlation(self, symbols: List[str],
                                    timeframe: str = None,
                                    start_time: Optional[datetime] = None,
                                    end_time: Optional[datetime] = None,
                                    method: str = None) -> pd.DataFrame:
        """
           Обчислює кореляцію цін закриття для заданих криптовалютних символів.

           Дані отримуються з бази даних за вказаним таймфреймом і періодом часу.
           Кореляційна матриця зберігається в базі даних.

           Parameters:
               symbols (List[str]): Список символів криптовалют.
               timeframe (str, optional): Таймфрейм даних. Якщо не вказано, використовується конфігурація.
               start_time (datetime, optional): Початкова дата. Якщо не вказано, використовується lookback період.
               end_time (datetime, optional): Кінцева дата. Якщо не вказано, використовується поточний час.
               method (str, optional): Метод обчислення кореляції (наприклад, 'pearson', 'kendall', 'spearman').

           Returns:
               pd.DataFrame: Матриця кореляції цін.
           """
        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = self.config['default_timeframe']
        method = self.config['default_correlation_method']

        self.logger.info(f"Розрахунок кореляції цін для {len(symbols)} символів з таймфреймом {timeframe}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            self.logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання цінових даних для всіх символів з бази даних замість Binance API
            price_data = {}
            for symbol in symbols:
                self.logger.debug(f"Отримання даних для {symbol}")
                # Використовуємо db_manager.get_klines замість binance_client.get_historical_prices
                price_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з цінами закриття для всіх символів
            df = pd.DataFrame()
            for symbol, data in price_data.items():
                data['open_time'] = pd.to_datetime(data['open_time'], unit='s')  # або unit='ms' — перевір формат
                data.set_index('open_time', inplace=True)
                df[symbol] = data['close']

            # Перевірка на відсутні дані
            if df.isnull().values.any():
                missing_count = df.isnull().sum().sum()
                self.logger.warning(f"Виявлено {missing_count} відсутніх значень. Заповнення методом forward fill")
                df = df.fillna(method='ffill')

            # Розрахунок матриці кореляції
            correlation_matrix = df.corr(method=method)

            self.logger.info(f"Матриця кореляції цін успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='price',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Помилка при розрахунку кореляції цін: {str(e)}")
            raise

    def calculate_volume_correlation(self, symbols: List[str],
                                     timeframe: str = None,
                                     start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None,
                                     method: str = None) -> pd.DataFrame:
        """
            Обчислює кореляцію об'ємів торгівлі для заданих криптовалютних символів.

            Дані про об'єми отримуються з бази даних. При необхідності очищаються від викидів.
            Результати кореляції зберігаються в базі даних.

            Parameters:
                symbols (List[str]): Список символів криптовалют.
                timeframe (str, optional): Таймфрейм даних. Якщо не вказано, використовується конфігурація.
                start_time (datetime, optional): Початкова дата. Якщо не вказано, використовується lookback період.
                end_time (datetime, optional): Кінцева дата. Якщо не вказано, використовується поточний час.
                method (str, optional): Метод обчислення кореляції (наприклад, 'pearson', 'kendall', 'spearman').

            Returns:
                pd.DataFrame: Матриця кореляції об'ємів.
            """
        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = timeframe or self.config['default_timeframe']
        method = method or self.config['default_correlation_method']

        self.logger.info(f"Розрахунок кореляції об'ємів торгівлі для {len(symbols)} символів з таймфреймом {timeframe}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            self.logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання даних про об'єми торгівлі для всіх символів з бази даних
            volume_data = {}
            for symbol in symbols:
                self.logger.debug(f"Отримання даних про об'єми для {symbol}")
                volume_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з об'ємами торгівлі для всіх символів
            df = pd.DataFrame()
            for symbol, data in volume_data.items():
                # Конвертація до float відразу під час створення датафрейму
                df[symbol] = data['volume'].astype(float)

            # Перевірка на відсутні значення
            if df.isnull().values.any():
                missing_count = df.isnull().sum().sum()
                self.logger.warning(f"Виявлено {missing_count} відсутніх значень об'єму. Заповнення методом forward fill")
                df = df.fillna(method='ffill')

            # Фільтрація даних для усунення викидів
            for column in df.columns:
                mean_val = df[column].mean()
                std_val = df[column].std()
                upper_limit = mean_val + 3 * std_val

                # Заміна значних викидів середнім значенням
                outliers_mask = df[column] > upper_limit
                if outliers_mask.any():
                    outlier_count = outliers_mask.sum()
                    self.logger.warning(
                        f"Виявлено {outlier_count} викидів об'єму для {column}. Заміна значеннями за медіаною")
                    df.loc[outliers_mask, column] = df[column].median()

            # Розрахунок матриці кореляції
            correlation_matrix = df.corr(method=method)

            self.logger.info(f"Матриця кореляції об'ємів торгівлі успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='volume',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Помилка при розрахунку кореляції об'ємів торгівлі: {str(e)}")
            raise

    def calculate_returns_correlation(self, symbols: List[str],
                                      timeframe: str = None,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      period: int = 1,
                                      method: str = None) -> pd.DataFrame:
        """
           Обчислює матрицю кореляції доходності (відсоткових змін) для заданих криптовалютних символів.

           Дані беруться з бази даних, перетворюються на доходності з вказаним періодом і використовуються
           для побудови матриці кореляції.

           Args:
               symbols (List[str]): Список криптовалютних символів (наприклад, ["BTC", "ETH"]).
               timeframe (str, optional): Таймфрейм даних (наприклад, "1d"). Якщо не вказано, використовується значення з конфігурації.
               start_time (datetime, optional): Початкова дата періоду. Якщо не вказано, використовується значення lookback з конфігурації.
               end_time (datetime, optional): Кінцева дата періоду. Якщо не вказано, береться поточна дата.
               period (int, optional): Період для обчислення відсоткових змін (доходності). За замовчуванням 1.
               method (str, optional): Метод розрахунку кореляції ('pearson', 'kendall', 'spearman').

           Returns:
               pd.DataFrame: Матриця кореляції доходності між криптовалютами.
           """
        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = timeframe or self.config['default_timeframe']
        method = method or self.config['default_correlation_method']

        self.logger.info(
            f"Розрахунок кореляції доходності для {len(symbols)} символів з таймфреймом {timeframe} та періодом {period}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            self.logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання цінових даних для всіх символів з бази даних
            price_data = {}
            for symbol in symbols:
                self.logger.debug(f"Отримання даних для {symbol}")
                # Використовуємо db_manager.get_klines замість binance_client.get_historical_prices
                price_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з цінами закриття для всіх символів
            df = pd.DataFrame()
            for symbol, data in price_data.items():
                data['open_time'] = pd.to_datetime(data['open_time'], unit='s')  # або unit='ms' — перевір формат
                data.set_index('open_time', inplace=True)
                df[symbol] = data['close']

            # Перевірка на відсутні дані
            if df.isnull().values.any():
                missing_count = df.isnull().sum().sum()
                self.logger.warning(f"Виявлено {missing_count} відсутніх значень. Заповнення методом forward fill")
                df = df.fillna(method='ffill')

            # Розрахунок доходності
            returns_df = df.pct_change(period)

            # Видалення перших рядків, які містять NaN через обчислення доходності
            returns_df = returns_df.iloc[period:]

            # Розрахунок матриці кореляції доходності
            correlation_matrix = returns_df.corr(method=method)

            self.logger.info(f"Матриця кореляції доходності успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='returns',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Помилка при розрахунку кореляції доходності: {str(e)}")
            raise

    def calculate_volatility_correlation(self, symbols: List[str],
                                         timeframe: str = None,
                                         start_time: Optional[datetime] = None,
                                         end_time: Optional[datetime] = None,
                                         window: int = None,
                                         method: str = None) -> pd.DataFrame:
        """
            Обчислює матрицю кореляції волатильності для заданих криптовалютних символів.

            Волатильність оцінюється як ковзне стандартне відхилення доходності. Дані беруться з бази даних,
            обробляються з урахуванням ковзного вікна і використовуються для обчислення кореляції.

            Args:
                symbols (List[str]): Список криптовалютних символів (наприклад, ["BTC", "ETH"]).
                timeframe (str, optional): Таймфрейм даних (наприклад, "1h"). Якщо не вказано, використовується значення з конфігурації.
                start_time (datetime, optional): Початкова дата періоду. Якщо не вказано, використовується значення lookback з конфігурації.
                end_time (datetime, optional): Кінцева дата періоду. Якщо не вказано, береться поточна дата.
                window (int, optional): Розмір ковзного вікна для розрахунку волатильності. Якщо не вказано, береться з конфігурації.
                method (str, optional): Метод розрахунку кореляції ('pearson', 'kendall', 'spearman').

            Returns:
                pd.DataFrame: Матриця кореляції волатильності між криптовалютами.
            """
        # Використання значень за замовчуванням з конфігурації, якщо не вказані
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']
        method = method or self.config['default_correlation_method']

        self.logger.info(
            f"Розрахунок кореляції волатильності для {len(symbols)} символів з таймфреймом {timeframe} та вікном {window}")

        # Встановлення кінцевого часу як поточний, якщо не вказано
        end_time = end_time or datetime.now()

        # Якщо початковий час не вказано, використовуємо значення за замовчуванням
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)
            self.logger.debug(f"Використання періоду за замовчуванням: {lookback_days} днів")

        try:
            # Отримання цінових даних для всіх символів з бази даних
            price_data = {}
            for symbol in symbols:
                self.logger.debug(f"Отримання даних для {symbol}")
                # Використовуємо db_manager.get_klines замість binance_client.get_historical_prices
                price_data[symbol] = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Створення датафрейму з цінами закриття для всіх символів
            price_df = pd.DataFrame()
            for symbol, data in price_data.items():
                data['open_time'] = pd.to_datetime(data['open_time'], unit='s')  # або unit='ms' — перевір формат
                data.set_index('open_time', inplace=True)
                price_df[symbol] = data['close']

            # Перевірка на відсутні дані
            if price_df.isnull().values.any():
                missing_count = price_df.isnull().sum().sum()
                self.logger.warning(f"Виявлено {missing_count} відсутніх значень. Заповнення методом forward fill")
                price_df = price_df.fillna(method='ffill')

            # Розрахунок доходності
            returns_df = price_df.pct_change()
            returns_df = returns_df.iloc[1:]  # Видалення першого рядка, який містить NaN

            # Розрахунок волатильності (стандартне відхилення доходності)
            volatility_df = pd.DataFrame()
            for symbol in symbols:
                volatility_df[symbol] = returns_df[symbol].rolling(window=window).std()

            # Видалення початкових рядків, які містять NaN через обчислення ковзного вікна
            volatility_df = volatility_df.iloc[window - 1:]

            # Розрахунок матриці кореляції волатильності
            correlation_matrix = volatility_df.corr(method=method)

            self.logger.info(f"Матриця кореляції волатильності успішно розрахована з використанням методу {method}")

            # Збереження результатів у базу даних
            self.save_correlation_to_db(
                correlation_matrix=correlation_matrix,
                correlation_type='volatility',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                method=method
            )

            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Помилка при розрахунку кореляції волатильності: {str(e)}")
            raise

    def get_correlated_pairs(self, correlation_matrix: pd.DataFrame,
                             threshold: float = None) -> List[Tuple[str, str, float]]:
        """
           Визначає пари активів із сильною позитивною кореляцією, що перевищує заданий поріг.

           Перебирає лише верхній трикутник матриці, щоб уникнути дублювання та самокореляцій.

           Args:
               correlation_matrix (pd.DataFrame): Квадратна матриця кореляцій між активами.
               threshold (float, optional): Поріг для сильної кореляції. Якщо не задано — береться з конфігурації.

           Returns:
               List[Tuple[str, str, float]]: Список кортежів у форматі (symbol1, symbol2, correlation),
               відсортованих за спаданням коефіцієнта кореляції.
           """
        # Use default threshold from config if not specified
        threshold = threshold or self.config['correlation_threshold']

        # Initialize results list
        correlated_pairs = []

        # Get the symbols (assuming they are both column and index names)
        symbols = correlation_matrix.columns.tolist()

        # Iterate through upper triangle of correlation matrix
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):  # Start from i+1 to avoid duplicates and self-correlations
                symbol1, symbol2 = symbols[i], symbols[j]
                correlation = correlation_matrix.loc[symbol1, symbol2]

                # Check if correlation is above threshold
                if correlation >= threshold:
                    correlated_pairs.append((symbol1, symbol2, correlation))

        # Sort pairs by correlation value in descending order
        correlated_pairs.sort(key=lambda x: x[2], reverse=True)

        self.logger.info(f"Found {len(correlated_pairs)} highly correlated pairs with threshold {threshold}")
        return correlated_pairs

    def get_anticorrelated_pairs(self, correlation_matrix: pd.DataFrame,
                                 threshold: float = None) -> List[Tuple[str, str, float]]:
        """
           Визначає пари активів з сильною негативною кореляцією, що менше або дорівнює порогу.

           Перебирає лише верхній трикутник матриці, щоб уникнути дублювання та самокореляцій.

           Args:
               correlation_matrix (pd.DataFrame): Квадратна матриця кореляцій між активами.
               threshold (float, optional): Поріг для сильної антикореляції (наприклад, -0.8). Якщо не задано — береться з конфігурації.

           Returns:
               List[Tuple[str, str, float]]: Список кортежів у форматі (symbol1, symbol2, correlation),
               відсортованих за зростанням (найбільш негативна кореляція — перша).
           """
        # Use default threshold from config if not specified
        threshold = threshold or self.config['anticorrelation_threshold']

        # Initialize results list
        anticorrelated_pairs = []

        # Get the symbols (assuming they are both column and index names)
        symbols = correlation_matrix.columns.tolist()

        # Iterate through upper triangle of correlation matrix
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):  # Start from i+1 to avoid duplicates and self-correlations
                symbol1, symbol2 = symbols[i], symbols[j]
                correlation = correlation_matrix.loc[symbol1, symbol2]

                # Check if correlation is below threshold (negative correlation)
                if correlation <= threshold:
                    anticorrelated_pairs.append((symbol1, symbol2, correlation))

        # Sort pairs by correlation value in ascending order (most negative first)
        anticorrelated_pairs.sort(key=lambda x: x[2])

        self.logger.info(f"Found {len(anticorrelated_pairs)} highly anti-correlated pairs with threshold {threshold}")
        return anticorrelated_pairs

    def calculate_rolling_correlation(self, symbol1: str, symbol2: str,
                                      timeframe: str = None,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      window: int = None,
                                      method: str = None) -> pd.Series:
        """
            Обчислює ковзну (rolling) кореляцію між двома активами за вказаний період часу.

            Дані беруться з бази, вирівнюються у часі та перетворюються у доходності, після чого
            обчислюється ковзна кореляція.

            Args:
                symbol1 (str): Перший символ (наприклад, "BTC").
                symbol2 (str): Другий символ (наприклад, "ETH").
                timeframe (str, optional): Таймфрейм (наприклад, "1h"). Якщо не вказано — береться з конфігурації.
                start_time (datetime, optional): Початок періоду. Якщо не вказано — визначається з урахуванням `window` і буфера.
                end_time (datetime, optional): Кінець періоду. Якщо не вказано — береться поточний час.
                window (int, optional): Розмір ковзного вікна. Якщо не вказано — береться з конфігурації.
                method (str, optional): Метод кореляції (тільки 'pearson' наразі використовується).

            Returns:
                pd.Series: Серія ковзних коефіцієнтів кореляції по часовій шкалі.
            """
        # Use default values from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']
        method = method or self.config['default_correlation_method']

        # Set default time range if not specified
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            # Add extra data to accommodate the window size
            start_time = end_time - timedelta(days=window + 30)  # +30 days buffer

        self.logger.info(f"Calculating rolling correlation between {symbol1} and {symbol2} with window={window}")

        try:
            # Fetch price data for both symbols
            df1 = self.db_manager.get_klines(
                symbol=symbol1,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            df2 = self.db_manager.get_klines(
                symbol=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Extract close prices
            price1 = df1['close']
            price2 = df2['close']

            # Align the time series (handle missing data points)
            aligned_prices = pd.DataFrame({
                symbol1: price1,
                symbol2: price2
            })

            # Calculate percent changes (returns)
            returns = aligned_prices.pct_change().dropna()

            # Calculate rolling correlation (method='pearson' only)
            rolling_corr = returns[symbol1].rolling(window=window).corr(returns[symbol2])

            self.logger.info(f"Successfully calculated rolling correlation with {len(rolling_corr)} data points")
            return rolling_corr

        except Exception as e:
            self.logger.error(f"Error calculating rolling correlation: {str(e)}")
            raise

    def detect_correlation_breakdowns(self, symbol1: str, symbol2: str,
                                      timeframe: str = None,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      window: int = None,
                                      threshold: float = None) -> List[datetime]:
        """
           Виявляє моменти значного розриву кореляції між двома активами.

           Метод обчислює ковзну кореляцію, виявляє різкі зміни, що перевищують поріг,
           і зберігає інформацію про ці розриви у базу даних.

           Args:
               symbol1 (str): Перший символ (наприклад, "ETH").
               symbol2 (str): Другий символ (наприклад, "BTC").
               timeframe (str, optional): Таймфрейм, наприклад "1d". Якщо не задано — використовується за замовчуванням.
               start_time (datetime, optional): Початкова дата для аналізу.
               end_time (datetime, optional): Кінцева дата для аналізу.
               window (int, optional): Розмір ковзного вікна.
               threshold (float, optional): Поріг для зміни кореляції, яка вважається розривом.

           Returns:
               List[datetime]: Список міток часу, де виявлено розрив кореляції.
           """
        # Use default values from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']
        threshold = threshold or self.config['breakdown_threshold']

        self.logger.info(f"Detecting correlation breakdowns between {symbol1} and {symbol2} with threshold {threshold}")

        try:
            # Calculate rolling correlation between the two symbols
            rolling_corr = self.calculate_rolling_correlation(
                symbol1=symbol1,
                symbol2=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                window=window
            )

            # Calculate absolute changes in correlation
            correlation_changes = rolling_corr.diff().abs()

            # Identify points where correlation changes more than the threshold
            breakdown_points = correlation_changes[correlation_changes > threshold].index.tolist()

            self.logger.info(f"Found {len(breakdown_points)} correlation breakdown points")

            # Save breakdown data to database
            breakdown_data = []
            for point in breakdown_points:
                correlation_before = rolling_corr.loc[rolling_corr.index < point].iloc[-1] if not rolling_corr.loc[
                    rolling_corr.index < point].empty else None
                correlation_after = rolling_corr.loc[point]
                change_magnitude = correlation_changes.loc[point]

                breakdown_data.append({
                    'timestamp': pd.to_datetime(point).to_pydatetime(),  # 🔧 конвертація до datetime.datetime
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'correlation_before': correlation_before,
                    'correlation_after': correlation_after,
                    'change_magnitude': change_magnitude
                })

            # Save to correlation_breakdowns table
            if breakdown_data:
                for point_data in breakdown_data:
                    self.db_manager.save_correlation_breakdown(
                        breakdown_time=point_data['timestamp'],
                        symbol1=point_data['symbol1'],
                        symbol2=point_data['symbol2'],
                        correlation_before=point_data['correlation_before'],
                        correlation_after=point_data['correlation_after'],
                        method=self.config.get('default_correlation_method', 'pearson'),
                        threshold=threshold,
                        timeframe=timeframe,
                        window_size=window
                    )
                self.logger.debug(f"Saved {len(breakdown_data)} breakdown points to database")

            return breakdown_points

        except Exception as e:
            self.logger.error(f"Error detecting correlation breakdowns: {str(e)}")
            raise

    def calculate_market_beta(self, symbol: str, market_symbol: str = 'BTC',
                              timeframe: str = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              window: int = None) -> Union[float, pd.Series]:
        """
           Обчислює бета-коефіцієнт активу відносно ринку (наприклад, BTC).

           Якщо задано `window`, обчислюється ковзне бета; інакше — загальне бета за весь період.
           Результати зберігаються в базу даних.

           Args:
               symbol (str): Символ активу (наприклад, "ETH").
               market_symbol (str, optional): Символ ринку для порівняння. За замовчуванням — "BTC".
               timeframe (str, optional): Таймфрейм для аналізу (наприклад, "1d").
               start_time (datetime, optional): Початок періоду аналізу.
               end_time (datetime, optional): Кінець періоду аналізу.
               window (int, optional): Розмір ковзного вікна для обчислення rolling beta.

           Returns:
               Union[float, pd.Series]: Загальний бета-коефіцієнт або серія ковзних значень бета.
           """
        # Use default values from config if not specified
        timeframe = timeframe or self.config['default_timeframe']
        window = window or self.config['default_correlation_window']

        self.logger.info(f"Calculating market beta for {symbol} relative to {market_symbol}")

        # Set default time range if not specified
        end_time = end_time or datetime.now()
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)

        try:
            # Fetch price data for asset and market benchmark
            asset_data = self.db_manager.get_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            market_data = self.db_manager.get_klines(
                symbol=market_symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Calculate returns and convert to float immediately
            asset_prices = asset_data['close'].astype(float)
            market_prices = market_data['close'].astype(float)

            # Align time series
            df = pd.DataFrame({
                'asset': asset_prices,
                'market': market_prices
            })

            # Calculate returns
            returns = df.pct_change().dropna()

            # If window is specified, calculate rolling beta
            if window:
                # Calculate rolling covariance and market variance
                rolling_cov = returns['asset'].rolling(window=window).cov(returns['market'])
                rolling_market_var = returns['market'].rolling(window=window).var()

                # Calculate rolling beta
                rolling_beta = rolling_cov / rolling_market_var

                # Save beta time series to database
                beta_records = []
                for timestamp, beta_value in rolling_beta.items():
                    if not np.isnan(beta_value) and not np.isinf(beta_value):
                        beta_records.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'market_symbol': market_symbol,
                            'beta': float(beta_value),  # Ensure beta is float
                            'timeframe': timeframe,
                            'window': window
                        })

                # Save to beta_time_series table
                if beta_records:
                    timestamps = [record['timestamp'] for record in beta_records]
                    beta_values = [record['beta'] for record in beta_records]

                    # Конвертуємо pd.Timestamp у datetime.datetime
                    timestamps = [ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts for ts in timestamps]

                    self.db_manager.save_beta_time_series(
                        symbol=symbol,
                        market_symbol=market_symbol,
                        timestamps=timestamps,
                        beta_values=beta_values,
                        timeframe=timeframe,
                        window_size=window
                    )

                    self.logger.debug(f"Saved {len(beta_records)} beta values to database")

                return rolling_beta
            else:
                # Calculate overall beta
                beta = np.cov(returns['asset'], returns['market'])[0, 1] / np.var(returns['market'])

                # Save to market_betas table
                beta_record = {
                    'symbol': symbol,
                    'market_symbol': market_symbol,
                    'beta': float(beta),  # Ensure beta is float
                    'timeframe': timeframe,
                    'start_time': start_time,
                    'end_time': end_time,
                }

                self.db_manager.save_market_beta([beta_record])
                self.logger.info(f"Saved beta value {beta} for {symbol} to database")

                return beta

        except Exception as e:
            self.logger.error(f"Error calculating market beta for {symbol}: {str(e)}")
            raise


    def save_correlation_to_db(self, correlation_matrix: pd.DataFrame,
                               correlation_type: str,
                               timeframe: str,
                               start_time: datetime,
                               end_time: datetime,
                               method: str) -> bool:
        """
            Зберігає матрицю кореляції та пари з високою кореляцією до бази даних.

            Створює унікальний ідентифікатор матриці, серіалізує дані, обчислює пари з абсолютною кореляцією вище порогу
            та зберігає ці дані в базу.

            Args:
                correlation_matrix (pd.DataFrame): Матриця кореляції.
                correlation_type (str): Тип кореляції (наприклад, "pearson").
                timeframe (str): Таймфрейм (наприклад, "1d").
                start_time (datetime): Початок аналізованого періоду.
                end_time (datetime): Кінець аналізованого періоду.
                method (str): Метод обчислення кореляції (наприклад, "pearson").

            Returns:
                bool: Повертає True, якщо збереження успішне; інакше False.
            """
        # Use default method from config if not specified
        method = method or self.config['default_correlation_method']

        self.logger.info(f"Saving {correlation_type} correlation matrix to database")

        try:
            # Create a record for the correlation matrix
            matrix_id = f"{correlation_type}_{timeframe}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}_{method}"

            # Prepare data for the correlation_matrices table
            matrix_data = {
                'matrix_id': matrix_id,
                'correlation_type': correlation_type,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time,
                'method': method,
                'analysis_time': datetime.now(),
                'symbols': correlation_matrix.columns.tolist(),
                'matrix_json': correlation_matrix.to_json()
            }

            # Save to correlation_matrices table
            self.db_manager.save_correlation_matrix(
                correlation_matrix.columns.tolist(),
                correlation_type,
                timeframe,
                start_time,
                end_time=end_time,
                method='pearson'
            )
            # Extract and save highly correlated pairs
            symbols = correlation_matrix.columns.tolist()
            pairs_data = []

            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    correlation = correlation_matrix.loc[symbol1, symbol2]

                    if abs(correlation) >= self.config['correlation_threshold']:
                        pair_data = {
                            'matrix_id': matrix_id,
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation_value': correlation,  # Changed from 'correlation'
                            'correlation_type': correlation_type,
                            'timeframe': timeframe,
                            'start_time': start_time,  # Added
                            'end_time': end_time,  # Added
                            'method': method,  # Added
                            'analysis_time': datetime.now()
                        }
                        pairs_data.append(pair_data)

            # Save to correlated_pairs table
            if pairs_data:
                self.db_manager.insert_correlated_pair(pairs_data,method='pearson')
                self.logger.debug(f"Saved {len(pairs_data)} correlated pairs to database")

            self.logger.info(f"Successfully saved correlation matrix with ID {matrix_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving correlation to database: {str(e)}")
            return False

    def load_correlation_from_db(self, correlation_type: str,
                                 timeframe: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 method: str = None) -> Optional[pd.DataFrame]:
        """
           Завантажує збережену матрицю кореляції з бази даних за заданим періодом та параметрами.

           Формує унікальний `matrix_id` для доступу до запису, зчитує JSON-рядок з бази даних і десеріалізує його в DataFrame.

           Args:
               correlation_type (str): Тип кореляції (наприклад, "pearson", "spearman").
               timeframe (str): Таймфрейм, за яким була побудована матриця (наприклад, "1d").
               start_time (datetime): Початок періоду аналізу.
               end_time (datetime): Кінець періоду аналізу.
               method (str, optional): Метод обчислення кореляції. Якщо не вказано — використовується значення з конфігурації.

           Returns:
               Optional[pd.DataFrame]: Відновлена матриця кореляції або `None`, якщо дані не знайдено чи виникла помилка.
           """
        # Use default method from config if not specified
        method = method or self.config['default_correlation_method']

        self.logger.info(f"Loading {correlation_type} correlation matrix from database")

        try:
            # Create the matrix ID for lookup
            matrix_id = f"{correlation_type}_{timeframe}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}_{method}"

            # Load from correlation_matrices table
            matrix_data = self.db_manager.get_correlation_matrix(matrix_id)

            if matrix_data:
                # Convert JSON string back to DataFrame
                matrix_json = matrix_data.get('matrix_json')
                if matrix_json:
                    correlation_matrix = pd.read_json(matrix_json)
                    self.logger.info(f"Successfully loaded correlation matrix with ID {matrix_id}")
                    return correlation_matrix

            self.logger.warning(f"No correlation matrix found with ID {matrix_id}")
            return None

        except Exception as e:
            self.logger.error(f"Error loading correlation from database: {str(e)}")
            return None

    def correlation_time_series(self, symbols_pair: Tuple[str, str],
                                correlation_window: int = None,
                                lookback_days: int = None,
                                timeframe: str = None) -> pd.Series:
        """
           Обчислює або завантажує з бази даних часовий ряд ковзної кореляції між двома символами.

           Якщо дані вже є в базі — використовуються вони. Інакше кореляція обчислюється та зберігається.

           Args:
               symbols_pair (Tuple[str, str]): Пара символів для аналізу кореляції (symbol1, symbol2).
               correlation_window (int, optional): Розмір вікна для ковзної кореляції. Якщо не задано, береться з конфігурації.
               lookback_days (int, optional): Період зворотного перегляду в днях для аналізу.
               timeframe (str, optional): Таймфрейм (наприклад, "1d", "1h").

           Returns:
               pd.Series: Часовий ряд ковзної кореляції між активами.
           """
        # Use default values from config if not specified
        correlation_window = correlation_window or self.config['default_correlation_window']
        lookback_days = lookback_days or self.config['default_lookback_days']
        timeframe = timeframe or self.config['default_timeframe']

        symbol1, symbol2 = symbols_pair
        self.logger.info(f"Calculating correlation time series between {symbol1} and {symbol2}")

        # Set time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        try:
            # First check if we already have this data in the database
            correlation_series = self.db_manager.get_correlation_time_series(
                symbol1=symbol1,
                symbol2=symbol2,
                start_time=start_time,
                end_time=end_time,
                timeframe=timeframe,
                window=correlation_window
            )

            if correlation_series is not None and not correlation_series.empty:
                self.logger.info(f"Found existing correlation time series in database")
                return correlation_series

            # If not found in database, calculate it
            self.logger.info(f"Calculating new correlation time series")

            # Calculate rolling correlation
            rolling_corr = self.calculate_rolling_correlation(
                symbol1=symbol1,
                symbol2=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                window=correlation_window
            )

            # Save correlation time series to database
            correlation_data = []
            for timestamp, corr_value in rolling_corr.items():
                if not np.isnan(corr_value):
                    correlation_data.append({
                        'timestamp': timestamp,
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': corr_value,
                        'timeframe': timeframe,
                        'window': correlation_window
                    })

            # Save to correlation_time_series table
            if correlation_data:
                self.db_manager.save_correlation_time_series(correlation_data,method='pearson')
                self.logger.debug(f"Saved {len(correlation_data)} correlation values to database")

            return rolling_corr

        except Exception as e:
            self.logger.error(f"Error calculating correlation time series: {str(e)}")
            raise

    def find_leading_indicators(self, target_symbol: str,
                                candidate_symbols: List[str],
                                lag_periods: List[int] = None,
                                timeframe: str = None,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Dict[int, float]]:
        """
            Аналізує список кандидатів на роль лідерів для заданого цільового символу за лагами у дохідності.

            Для кожного кандидата обчислюється кореляція між лагованими доходностями та доходністю цільового активу.
            Результати зберігаються в базу даних, якщо є достатня кількість валідних даних.

            Args:
                target_symbol (str): Символ, для якого шукаються лідери.
                candidate_symbols (List[str]): Список символів-кандидатів.
                lag_periods (List[int], optional): Перелік лагів для перевірки (в кількості кроків таймфрейму).
                timeframe (str, optional): Таймфрейм для аналізу (наприклад, "1d").
                start_time (datetime, optional): Початковий час аналізу.
                end_time (datetime, optional): Кінцевий час аналізу.

            Returns:
                Dict[str, Dict[int, float]]: Словник, де ключ — символ-кандидат, а значення — словник з лагами та відповідними коефіцієнтами кореляції.
            """
        # Use default values from config if not specified
        lag_periods = lag_periods or self.config['default_lag_periods']
        timeframe = timeframe or self.config['default_timeframe']

        self.logger.info(f"Finding leading indicators for {target_symbol} among {len(candidate_symbols)} candidates")

        # Set default time range if not specified
        end_time = end_time or datetime.now()
        if start_time is None:
            lookback_days = self.config['default_lookback_days']
            start_time = end_time - timedelta(days=lookback_days)

        try:
            # Fetch target data
            target_data = self.db_manager.get_klines(
                symbol=target_symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Calculate target returns
            target_prices = target_data['close']
            target_returns = target_prices.pct_change().dropna()

            # Dictionary to store results
            leading_indicators = {}
            leading_indicators_records = []

            # Test each candidate symbol
            for symbol in candidate_symbols:
                if symbol == target_symbol:
                    continue

                self.logger.debug(f"Testing {symbol} as a leading indicator")

                # Fetch candidate data
                candidate_data = self.db_manager.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

                # Calculate candidate returns
                candidate_prices = candidate_data['close']
                candidate_returns = candidate_prices.pct_change().dropna()

                # Align time series
                common_index = target_returns.index.intersection(candidate_returns.index)
                if len(common_index) < max(lag_periods) + 10:  # Need enough data points
                    self.logger.warning(f"Insufficient aligned data points for {symbol}")
                    continue

                aligned_target = target_returns.loc[common_index]
                aligned_candidate = candidate_returns.loc[common_index]

                # Calculate lagged correlations
                lag_correlations = {}
                for lag in lag_periods:
                    # Shift candidate data forward to test if it leads target
                    lagged_candidate = aligned_candidate.shift(lag)

                    # Remove NaN values created by shifting
                    valid_indices = aligned_target.index.intersection(lagged_candidate.dropna().index)
                    if len(valid_indices) < 10:  # Need enough data points for correlation
                        continue

                    # Calculate correlation between target and lagged candidate
                    correlation = aligned_target.loc[valid_indices].corr(lagged_candidate.loc[valid_indices])
                    lag_correlations[lag] = correlation

                    # Add to records for database
                    leading_indicators_records.append({
                        'target_symbol': target_symbol,
                        'indicator_symbol': symbol,
                        'lag_period': lag,
                        'correlation_value': correlation,
                        'timeframe': timeframe,
                        'start_time': start_time,
                        'end_time': end_time,
                        'analysis_time': datetime.now(),
                        'method': 'pearson'
                    })

                # Only add to results if at least one lag period had a correlation
                if lag_correlations:
                    leading_indicators[symbol] = lag_correlations

            # Save to leading_indicators table
            if leading_indicators_records:
                self.db_manager.save_leading_indicators(leading_indicators_records)
                self.logger.debug(f"Saved {len(leading_indicators_records)} leading indicator records to database")

            return leading_indicators

        except Exception as e:
            self.logger.error(f"Error finding leading indicators: {str(e)}")
            raise

    def correlated_movement_prediction(self, symbol: str,
                                       correlated_symbols: List[str],
                                       prediction_horizon: int = None,
                                       timeframe: str = None) -> Dict[str, float]:
        """
            Прогнозує майбутній напрямок руху ціни заданого активу (symbol) на основі історичних прибутків
            скорельованих активів, використовуючи лінійну регресію.

            Parameters
            ----------
            symbol : str
                Основний символ (криптовалюта), для якого виконується прогноз.
            correlated_symbols : List[str]
                Список скорельованих символів, що використовуються як ознаки.
            prediction_horizon : int, optional
                Горизонт прогнозу у кількості періодів (наприклад, днів). Якщо не вказано — береться з конфігурації.
            timeframe : str, optional
                Таймфрейм (наприклад, '1d', '4h'). Якщо не вказано — використовується значення за замовчуванням.

            Returns
            -------
            Dict[str, Union[float, str, Dict[str, float], datetime]]
                Словник з результатами прогнозу, що включає:
                - symbol : символ активу
                - current_price : поточна ціна
                - predicted_return : передбачене відсоткове змінення ціни
                - predicted_price : передбачена ціна
                - prediction_horizon : горизонт прогнозу
                - prediction_direction : напрямок ("up" або "down")
                - confidence : довіра до прогнозу (0–1)
                - model_r2 : коефіцієнт детермінації R²
                - top_indicators : топ-5 ознак за вагомістю
                - prediction_timestamp : дата та час прогнозу

                У разі помилки повертає: {"error": "текст помилки"}
            """
        # Use default values from config if not specified
        prediction_horizon = prediction_horizon or self.config['default_prediction_horizon']
        timeframe = timeframe or self.config['default_timeframe']

        self.logger.info(
            f"Predicting movement for {symbol} using {len(correlated_symbols)} correlated symbols with horizon {prediction_horizon}")

        # Set default time range - we need more historical data for training
        end_time = datetime.now()
        lookback_days = self.config['default_lookback_days'] * 2  # Double the usual lookback for better training
        start_time = end_time - timedelta(days=lookback_days)

        try:
            # Fetch price data for target and all correlated symbols
            all_symbols = [symbol] + correlated_symbols
            price_data = {}
            for sym in all_symbols:
                price_data[sym] = self.db_manager.get_klines(
                    symbol=sym,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Extract close prices and convert to float
            close_prices = {}
            for sym, data in price_data.items():
                # Convert Decimal to float if needed
                close_prices[sym] = pd.Series(data['close']).astype(float)

            # Calculate returns
            returns = {}
            for sym, prices in close_prices.items():
                returns[sym] = prices.pct_change().dropna()

            # Create a dataframe with all returns
            all_returns = pd.DataFrame({sym: ret for sym, ret in returns.items()})

            # Handle any missing data
            if all_returns.isnull().values.any():
                missing_count = all_returns.isnull().sum().sum()
                self.logger.warning(f"Found {missing_count} missing values in returns. Filling with forward fill.")
                all_returns = all_returns.fillna(method='ffill')

            # Create prediction features by shifting correlated symbols' returns
            feature_df = pd.DataFrame()
            for i, corr_sym in enumerate(correlated_symbols):
                for lag in range(1, prediction_horizon + 1):
                    feature_name = f"{corr_sym}_lag_{lag}"
                    feature_df[feature_name] = all_returns[corr_sym].shift(lag)

            # Target is future return of the symbol we're trying to predict
            target = all_returns[symbol].shift(-prediction_horizon)

            # Combine features and target
            model_data = pd.concat([feature_df, target], axis=1)
            model_data.columns = list(feature_df.columns) + ['target']

            # Remove rows with NaN values
            model_data = model_data.dropna()

            if len(model_data) < 30:  # Need sufficient data for reliable prediction
                self.logger.warning(f"Insufficient data for prediction after preprocessing: {len(model_data)} rows")
                return {"error": "Insufficient data for prediction"}

            # Split data into training and testing sets
            train_size = int(len(model_data) * 0.8)
            train_data = model_data.iloc[:train_size]
            test_data = model_data.iloc[train_size:]

            # Train a simple linear regression model
            from sklearn.linear_model import LinearRegression

            # Ensure all data is float type
            X_train = train_data.drop('target', axis=1).astype(float)
            y_train = train_data['target'].astype(float)

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Evaluate on test set - ensure test data is also float type
            X_test = test_data.drop('target', axis=1).astype(float)
            y_test = test_data['target'].astype(float)

            test_score = model.score(X_test, y_test)
            self.logger.info(f"Prediction model R² score: {test_score:.4f}")

            # Get the most recent data for prediction
            latest_data = feature_df.iloc[-1:].astype(float)
            if latest_data.isnull().values.any():
                latest_data = latest_data.fillna(0)  # Handle any missing values in latest data

            # Make prediction for future movement
            predicted_return = float(model.predict(latest_data)[0])  # Ensure result is float

            # Get feature importances (coefficients)
            feature_importance = dict(zip(X_train.columns, model.coef_))

            # Sort features by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = dict(sorted_features[:5])  # Top 5 most important features

            # Calculate confidence metric based on historical accuracy
            from sklearn.metrics import mean_squared_error
            y_pred_test = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)

            # Normalize RMSE to get a confidence score between 0 and 1
            avg_abs_return = np.mean(np.abs(y_test))
            confidence = float(max(0, min(1, 1 - (rmse / (avg_abs_return * 2)))))  # Ensure result is float

            # Get current price for the symbol and ensure it's float
            current_price = float(close_prices[symbol].iloc[-1])

            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_return)

            # Prepare results
            results = {
                "symbol": symbol,
                "current_price": current_price,
                "predicted_return": predicted_return,
                "predicted_price": predicted_price,
                "prediction_horizon": prediction_horizon,
                "prediction_direction": "up" if predicted_return > 0 else "down",
                "confidence": confidence,
                "model_r2": float(test_score),  # Ensure R² is float
                "top_indicators": {k: float(v) for k, v in top_features.items()},  # Convert coefficients to float
                "prediction_timestamp": datetime.now()
            }

            # Save prediction to database
            prediction_record = {
                "target_symbol": symbol,
                "predicted_return": predicted_return,
                "prediction_horizon": prediction_horizon,
                "confidence": confidence,
                "model_type": "linear_regression",
                "prediction_time": datetime.now(),
                "timeframe": timeframe
                # Additional fields could be added if needed
            }

            # self.db_manager.save_prediction(prediction_record)

            self.logger.info(f"Successfully generated prediction for {symbol} with {prediction_horizon} periods horizon")
            return results

        except Exception as e:
            self.logger.error(f"Error in correlated movement prediction: {str(e)}")
            return {"error": str(e)}


    def analyze_market_regime_correlations(self, symbols: List[str],
                                           market_regimes: Dict[Tuple[datetime, datetime], str],
                                           timeframe: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
           Аналізує кореляції між символами у межах заданих ринкових режимів.
           Проводиться розрахунок кореляцій цін, прибутків, волатильності та об'ємів
           для кожного ринкового режиму окремо.

           Parameters
           ----------
           symbols : List[str]
               Список символів (наприклад, криптовалют), для яких виконується аналіз.
           market_regimes : Dict[Tuple[datetime, datetime], str]
               Словник, де ключ — це пара (start_time, end_time), а значення — назва режиму.
           timeframe : str, optional
               Таймфрейм для аналізу (наприклад, '1d', '1h'). Якщо не вказано — використовується з конфігурації.

           Returns
           -------
           Dict[str, Dict[str, pd.DataFrame]]
               Словник, який повертає для кожного режиму вкладені словники з типами кореляцій:
               regime_name → {
                   "price": pd.DataFrame,
                   "returns": pd.DataFrame,
                   "volatility": pd.DataFrame,
                   "volume": pd.DataFrame
               }

           Raises
           ------
           Exception
               У разі фатальної помилки під час обробки повертає виключення з логуванням.

           """
        # Use default timeframe from config if not specified
        timeframe = timeframe or self.config['default_timeframe']

        self.logger.info(
            f"Analyzing market regime correlations for {len(symbols)} symbols across {len(market_regimes)} regimes")

        # Dictionary to store results by regime
        regime_correlations = {}

        # Correlation types to calculate
        correlation_types = ['price', 'returns', 'volatility', 'volume']

        try:
            # Process each market regime
            for (regime_start, regime_end), regime_name in market_regimes.items():
                self.logger.info(f"Analyzing regime '{regime_name}' from {regime_start} to {regime_end}")

                # Skip regimes with invalid time ranges
                if regime_start >= regime_end:
                    self.logger.warning(f"Invalid time range for regime {regime_name}: {regime_start} to {regime_end}")
                    continue

                # Initialize regime entry in results dictionary if not exists
                if regime_name not in regime_correlations:
                    regime_correlations[regime_name] = {}

                # Calculate price correlation for this regime
                try:
                    price_corr = self.calculate_price_correlation(
                        symbols=symbols,
                        timeframe=timeframe,
                        start_time=regime_start,
                        end_time=regime_end
                    )
                    regime_correlations[regime_name]['price'] = price_corr
                    self.logger.debug(f"Calculated price correlation for regime {regime_name}")
                except Exception as e:
                    self.logger.warning(f"Error calculating price correlation for regime {regime_name}: {str(e)}")

                # Calculate returns correlation
                try:
                    returns_corr = self.calculate_returns_correlation(
                        symbols=symbols,
                        timeframe=timeframe,
                        start_time=regime_start,
                        end_time=regime_end
                    )
                    regime_correlations[regime_name]['returns'] = returns_corr
                    self.logger.debug(f"Calculated returns correlation for regime {regime_name}")
                except Exception as e:
                    self.logger.warning(f"Error calculating returns correlation for regime {regime_name}: {str(e)}")

                # Calculate volatility correlation
                try:
                    volatility_corr = self.calculate_volatility_correlation(
                        symbols=symbols,
                        timeframe=timeframe,
                        start_time=regime_start,
                        end_time=regime_end
                    )
                    regime_correlations[regime_name]['volatility'] = volatility_corr
                    self.logger.debug(f"Calculated volatility correlation for regime {regime_name}")
                except Exception as e:
                    self.logger.warning(f"Error calculating volatility correlation for regime {regime_name}: {str(e)}")

                # Calculate volume correlation
                try:
                    volume_corr = self.calculate_volume_correlation(
                        symbols=symbols,
                        timeframe=timeframe,
                        start_time=regime_start,
                        end_time=regime_end
                    )
                    regime_correlations[regime_name]['volume'] = volume_corr
                    self.logger.debug(f"Calculated volume correlation for regime {regime_name}")
                except Exception as e:
                    self.logger.warning(f"Error calculating volume correlation for regime {regime_name}: {str(e)}")

                # Calculate average correlations for this regime
                if 'returns' in regime_correlations[regime_name]:
                    returns_matrix = regime_correlations[regime_name]['returns']

                    # Get average correlation value (excluding self-correlations)
                    corr_values = []
                    for i in range(len(symbols)):
                        for j in range(i + 1, len(symbols)):
                            corr_values.append(returns_matrix.iloc[i, j])

                    avg_corr = sum(corr_values) / len(corr_values) if corr_values else 0
                    self.logger.info(f"Regime {regime_name} - Average correlation: {avg_corr:.4f}")

            # Save regime correlation analysis to database
            for regime_name, correlation_data in regime_correlations.items():
                for corr_type, corr_matrix in correlation_data.items():
                    # Create record
                    regime_start, regime_end = None, None
                    for (start, end), name in market_regimes.items():
                        if name == regime_name:
                            regime_start, regime_end = start, end
                            break

                    if regime_start and regime_end:
                        record = {
                            'regime_name': regime_name,
                            'correlation_type': corr_type,
                            'timeframe': timeframe,
                            'start_time': regime_start,
                            'end_time': regime_end,
                            'analysis_time': datetime.now(),
                            'symbols': symbols,
                            'matrix_json': corr_matrix.to_json()
                        }

                        # Assuming we have a method to save regime correlation data
                        if hasattr(self.db_manager, 'save_market_regime_correlation'):
                            self.db_manager.save_market_regime_correlation(record)
                            self.logger.debug(f"Saved {corr_type} correlation for regime {regime_name} to database")

            self.logger.info(f"Market regime correlation analysis complete for {len(regime_correlations)} regimes")
            return regime_correlations

        except Exception as e:
            self.logger.error(f"Error analyzing market regime correlations: {str(e)}")
            raise


def main():
    """
    Тестовий скрипт для демонстрації можливостей класу MarketCorrelation
    """
    print("Початок тестування MarketCorrelation...")

    # Ініціалізація об'єкта аналізатора кореляції
    mc = MarketCorrelation()

    # Список криптовалютних пар для аналізу
    symbols = ['BTC', 'ETH','SOL']
    print(f"Аналізуємо наступні символи: {', '.join(symbols)}")

    # Встановлення часових рамок для аналізу
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # дані за останні 30 днів
    print(f"Період аналізу: з {start_time.strftime('%Y-%m-%d')} по {end_time.strftime('%Y-%m-%d')}")

    # Встановлення таймфрейму
    timeframe = '1h'  # 1-годинний таймфрейм

    try:
        # 1. Розрахунок кореляції цін
        print("\n1. Розрахунок кореляції цін...")
        price_corr = mc.calculate_price_correlation(
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        print("Матриця кореляції цін:")
        print(price_corr.round(2))

        # 2. Знаходження високо корельованих пар активів
        print("\n2. Високо корельовані пари:")
        correlated_pairs = mc.get_correlated_pairs(price_corr, threshold=0.7)
        for symbol1, symbol2, corr in correlated_pairs:
            print(f"{symbol1} і {symbol2}: {corr:.4f}")

        # 3. Знаходження анти-корельованих пар активів
        print("\n3. Анти-корельовані пари:")
        anticorrelated_pairs = mc.get_anticorrelated_pairs(price_corr, threshold=-0.3)
        for symbol1, symbol2, corr in anticorrelated_pairs:
            print(f"{symbol1} і {symbol2}: {corr:.4f}")

        # 4. Розрахунок кореляції доходності
        print("\n4. Розрахунок кореляції доходності...")
        returns_corr = mc.calculate_returns_correlation(
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        print("Матриця кореляції доходності:")
        print(returns_corr.round(2))

        # 5. Розрахунок кореляції волатильності
        print("\n5. Розрахунок кореляції волатильності...")
        volatility_corr = mc.calculate_volatility_correlation(
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        print("Матриця кореляції волатильності:")
        print(volatility_corr.round(2))

        # 6. Розрахунок кореляції об'єму торгівлі
        print("\n6. Розрахунок кореляції об'єму торгівлі...")
        volume_corr = mc.calculate_volume_correlation(
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        print("Матриця кореляції об'єму торгівлі:")
        print(volume_corr.round(2))

        # 7. Вибір однієї пари для аналізу динамічної кореляції
        if correlated_pairs:
            symbol1, symbol2, _ = correlated_pairs[0]
            print(f"\n7. Аналіз динамічної кореляції між {symbol1} і {symbol2}...")

            # Розрахунок змінної кореляції з часом
            rolling_corr = mc.calculate_rolling_correlation(
                symbol1=symbol1,
                symbol2=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                window=24  # вікно у 24 години
            )

            print(f"Поточна кореляція: {rolling_corr.iloc[-1]:.4f}")

            # Виявлення зламів у кореляції
            print("\n8. Виявлення зламів кореляції...")
            breakdown_points = mc.detect_correlation_breakdowns(
                symbol1=symbol1,
                symbol2=symbol2,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                threshold=0.2  # суттєва зміна кореляції
            )

            print(f"Знайдено {len(breakdown_points)} точок зламу кореляції")
            if breakdown_points:
                for point in breakdown_points:
                    print(f"Злам кореляції на {point}")

        # 9. Розрахунок бети відносно BTC
        print("\n9. Розрахунок бети відносно BTC...")
        for symbol in symbols:
            if symbol == 'BTC':
                continue

            beta = mc.calculate_market_beta(
                symbol=symbol,
                market_symbol='BTC',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            if isinstance(beta, float):
                print(f"Бета для {symbol} відносно BTC: {beta:.4f}")

        # 10. Пошук ведучих індикаторів
        print("\n10. Пошук ведучих індикаторів для BTCUSDT...")
        other_symbols = [s for s in symbols if s != 'BTC']
        leading_indicators = mc.find_leading_indicators(
            target_symbol='BTC',
            candidate_symbols=other_symbols,
            lag_periods=[1, 2, 3, 6, 12, 24],  # лаги у годинах
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        print("Результати пошуку ведучих індикаторів:")
        for symbol, lags in leading_indicators.items():
            best_lag = max(lags.items(), key=lambda x: abs(x[1]))
            print(f"{symbol} на лагу {best_lag[0]} годин: кореляція {best_lag[1]:.4f}")

        # 11. Прогнозування руху на основі корельованих активів
        print("\n11. Прогнозування руху BTC на основі корельованих активів...")
        prediction = mc.correlated_movement_prediction(
            symbol='BTC',
            correlated_symbols=other_symbols,
            prediction_horizon=24,  # прогноз на 24 години
            timeframe=timeframe
        )

        print("Результати прогнозування:")
        for key, value in prediction.items():
            print(f"{key}: {value}")

        # 12. Аналіз кореляцій за різних ринкових режимів
        print("\n12. Аналіз кореляцій за різних ринкових режимів...")
        # Визначення ринкових режимів (приклад)
        regime_start = start_time
        regime_mid = start_time + (end_time - start_time) / 2

        market_regimes = {
            (regime_start, regime_mid): "Перша половина періоду",
            (regime_mid, end_time): "Друга половина періоду"
        }

        regime_correlations = mc.analyze_market_regime_correlations(
            symbols=symbols,
            market_regimes=market_regimes,
            timeframe=timeframe
        )

        print("Результати аналізу ринкових режимів:")
        for regime_name, corr_data in regime_correlations.items():
            if 'returns' in corr_data:
                returns_matrix = corr_data['returns']
                avg_corr = returns_matrix.values[np.triu_indices_from(returns_matrix.values, k=1)].mean()
                print(f"Режим '{regime_name}' - середня кореляція доходності: {avg_corr:.4f}")

        print("\nТестування MarketCorrelation завершено успішно!")

    except Exception as e:
        print(f"Помилка при тестуванні: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

