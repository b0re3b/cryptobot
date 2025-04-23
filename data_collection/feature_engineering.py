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

        self.logger.info("Створення ознак на основі експоненціально зваженого вікна...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

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

        # Словник підтримуваних функцій для EWM
        func_map = {
            'mean': 'mean',
            'std': 'std',
            'var': 'var',
        }

        # Перевіряємо, чи всі функції підтримуються
        unsupported_funcs = [f for f in functions if f not in func_map]
        if unsupported_funcs:
            self.logger.warning(f"Функції {unsupported_funcs} не підтримуються для EWM і будуть пропущені")
            functions = [f for f in functions if f in func_map]

        # Лічильник доданих ознак
        added_features_count = 0

        # Для кожного стовпця, span і функції створюємо нову ознаку
        for col in columns:
            for span in spans:
                # Перевіряємо наявність пропущених значень в стовпці
                if result_df[col].isna().any():
                    self.logger.warning(f"Стовпець {col} містить NaN значення, вони будуть заповнені")
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                # Створюємо об'єкт експоненціально зваженого вікна
                ewm_window = result_df[col].ewm(span=span, min_periods=1)

                for func_name in functions:
                    # Отримуємо функцію з мапінгу
                    func = func_map[func_name]

                    # Створюємо нову ознаку
                    feature_name = f"{col}_ewm_{span}_{func_name}"

                    # Застосовуємо функцію до EWM
                    result_df[feature_name] = getattr(ewm_window, func)()

                    added_features_count += 1
                    self.logger.debug(f"Створено ознаку {feature_name}")

        # Перевіряємо наявність NaN значень у нових ознаках
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    self.logger.debug(f"Заповнення NaN значень у стовпці {col}")
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                    # Якщо все ще є NaN (можливо на початку), заповнюємо медіаною
                    if result_df[col].isna().any():
                        result_df[col] = result_df[col].fillna(result_df[col].median())

        self.logger.info(f"Додано {added_features_count} ознак експоненціально зваженого вікна")

        return result_df

    def create_return_features(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:

        self.logger.info("Створення ознак прибутковості...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що price_column існує в даних
        if price_column not in result_df.columns:
            self.logger.error(f"Стовпець {price_column} не знайдено в даних")
            raise ValueError(f"Стовпець {price_column} не знайдено в даних")

        # Перевіряємо наявність пропущених значень у стовпці ціни
        if result_df[price_column].isna().any():
            self.logger.warning(f"Стовпець {price_column} містить NaN значення, вони будуть заповнені")
            result_df[price_column] = result_df[price_column].fillna(method='ffill').fillna(method='bfill')

        # Лічильник доданих ознак
        added_features_count = 0

        # Розрахунок процентної зміни для кожного періоду
        for period in periods:
            # Процентна зміна
            pct_change_name = f"return_{period}p"
            result_df[pct_change_name] = result_df[price_column].pct_change(periods=period)
            added_features_count += 1

            # Логарифмічна прибутковість
            log_return_name = f"log_return_{period}p"
            result_df[log_return_name] = np.log(result_df[price_column] / result_df[price_column].shift(period))
            added_features_count += 1

            # Абсолютна зміна
            abs_change_name = f"abs_change_{period}p"
            result_df[abs_change_name] = result_df[price_column].diff(periods=period)
            added_features_count += 1

            # Нормалізована зміна (Z-score над N періодами)
            z_score_period = min(period * 5, len(result_df))  # беремо більший період для розрахунку статистики
            if z_score_period > period * 2:  # перевіряємо, що маємо достатньо даних для нормалізації
                z_score_name = f"z_score_return_{period}p"
                rolling_mean = result_df[pct_change_name].rolling(window=z_score_period).mean()
                rolling_std = result_df[pct_change_name].rolling(window=z_score_period).std()
                result_df[z_score_name] = (result_df[pct_change_name] - rolling_mean) / rolling_std
                added_features_count += 1

        # Додаємо ознаку напрямку зміни ціни (бінарна класифікація)
        for period in periods:
            direction_name = f"direction_{period}p"
            result_df[direction_name] = np.where(result_df[f"return_{period}p"] > 0, 1, 0)
            added_features_count += 1

        # Заповнюємо NaN значення (особливо на початку часового ряду)
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    # Для ознак напрямку використовуємо 0 (нейтральне значення)
                    if col.startswith("direction_"):
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        # Для інших ознак використовуємо 0 або медіану
                        result_df[col] = result_df[col].fillna(0)

        self.logger.info(f"Додано {added_features_count} ознак прибутковості")

        return result_df

    def create_technical_features(self, data: pd.DataFrame,
                                  indicators: Optional[List[str]] = None) -> pd.DataFrame:

        self.logger.info("Створення технічних індикаторів...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що необхідні OHLCV колонки існують
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in result_df.columns]

        if missing_columns:
            self.logger.warning(
                f"Відсутні деякі OHLCV колонки: {missing_columns}. Деякі індикатори можуть бути недоступні.")

        # Визначаємо базовий набір індикаторів, якщо не вказано
        if indicators is None:
            indicators = [
                'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
                'stochastic', 'atr', 'adx', 'obv', 'roc', 'cci'
            ]
            self.logger.info(f"Використовується базовий набір індикаторів: {indicators}")

        # Лічильник доданих ознак
        added_features_count = 0

        # Індикатори на основі бібліотеки ta
        for indicator in indicators:
            try:
                # Прості ковзні середні
                if indicator == 'sma':
                    for window in [5, 10, 20, 50, 200]:
                        if 'close' in result_df.columns:
                            result_df[f'sma_{window}'] = ta.trend.sma_indicator(result_df['close'], window=window)
                            added_features_count += 1

                # Експоненціальні ковзні середні
                elif indicator == 'ema':
                    for window in [5, 10, 20, 50, 200]:
                        if 'close' in result_df.columns:
                            result_df[f'ema_{window}'] = ta.trend.ema_indicator(result_df['close'], window=window)
                            added_features_count += 1

                # Relative Strength Index
                elif indicator == 'rsi':
                    for window in [7, 14, 21]:
                        if 'close' in result_df.columns:
                            result_df[f'rsi_{window}'] = ta.momentum.rsi(result_df['close'], window=window)
                            added_features_count += 1

                # Moving Average Convergence Divergence
                elif indicator == 'macd':
                    if 'close' in result_df.columns:
                        macd = ta.trend.macd(result_df['close'], fast=12, slow=26, signal=9)
                        result_df['macd_line'] = macd.iloc[:, 0]
                        result_df['macd_signal'] = macd.iloc[:, 1]
                        result_df['macd_histogram'] = macd.iloc[:, 2]
                        added_features_count += 3

                # Bollinger Bands
                elif indicator == 'bollinger_bands':
                    for window in [20]:
                        if 'close' in result_df.columns:
                            result_df[f'bb_high_{window}'] = ta.volatility.bollinger_hband(result_df['close'],
                                                                                           window=window)
                            result_df[f'bb_mid_{window}'] = ta.volatility.bollinger_mavg(result_df['close'],
                                                                                         window=window)
                            result_df[f'bb_low_{window}'] = ta.volatility.bollinger_lband(result_df['close'],
                                                                                          window=window)
                            result_df[f'bb_width_{window}'] = (result_df[f'bb_high_{window}'] - result_df[
                                f'bb_low_{window}']) / result_df[f'bb_mid_{window}']
                            added_features_count += 4

                # Stochastic Oscillator
                elif indicator == 'stochastic':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        result_df['stoch_k'] = ta.momentum.stoch(result_df['high'], result_df['low'],
                                                                 result_df['close'])
                        result_df['stoch_d'] = ta.momentum.stoch_signal(result_df['high'], result_df['low'],
                                                                        result_df['close'])
                        added_features_count += 2

                # Average True Range
                elif indicator == 'atr':
                    for window in [14]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            result_df[f'atr_{window}'] = ta.volatility.average_true_range(result_df['high'],
                                                                                          result_df['low'],
                                                                                          result_df['close'],
                                                                                          window=window)
                            added_features_count += 1

                # Average Directional Index
                elif indicator == 'adx':
                    for window in [14]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            result_df[f'adx_{window}'] = ta.trend.adx(result_df['high'], result_df['low'],
                                                                      result_df['close'], window=window)
                            result_df[f'adx_pos_{window}'] = ta.trend.adx_pos(result_df['high'], result_df['low'],
                                                                              result_df['close'], window=window)
                            result_df[f'adx_neg_{window}'] = ta.trend.adx_neg(result_df['high'], result_df['low'],
                                                                              result_df['close'], window=window)
                            added_features_count += 3

                # On Balance Volume
                elif indicator == 'obv':
                    if all(col in result_df.columns for col in ['close', 'volume']):
                        result_df['obv'] = ta.volume.on_balance_volume(result_df['close'], result_df['volume'])
                        added_features_count += 1

                # Rate of Change
                elif indicator == 'roc':
                    for window in [5, 10, 20]:
                        if 'close' in result_df.columns:
                            result_df[f'roc_{window}'] = ta.momentum.roc(result_df['close'], window=window)
                            added_features_count += 1

                # Commodity Channel Index
                elif indicator == 'cci':
                    for window in [20]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            result_df[f'cci_{window}'] = ta.trend.cci(result_df['high'], result_df['low'],
                                                                      result_df['close'], window=window)
                            added_features_count += 1

                # Додаткові індикатори можна додати тут
                else:
                    self.logger.warning(f"Індикатор {indicator} не підтримується і буде пропущений")

            except Exception as e:
                self.logger.error(f"Помилка при розрахунку індикатора {indicator}: {str(e)}")

        # Заповнюємо NaN значення
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    self.logger.debug(f"Заповнення NaN значень у стовпці {col}")
                    # Заповнюємо NaN методом forward fill, потім backward fill
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                    # Якщо все ще є NaN, заповнюємо медіаною
                    if result_df[col].isna().any():
                        result_df[col] = result_df[col].fillna(result_df[col].median())

        self.logger.info(f"Додано {added_features_count} технічних індикаторів")

        return result_df

    def create_volatility_features(self, data: pd.DataFrame,
                                   price_column: str = 'close',
                                   window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:

        self.logger.info("Створення ознак волатильності...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що price_column існує в даних
        if price_column not in result_df.columns:
            self.logger.error(f"Стовпець {price_column} не знайдено в даних")
            raise ValueError(f"Стовпець {price_column} не знайдено в даних")

        # Перевіряємо наявність OHLC стовпців для розрахунку додаткових метрик волатильності
        has_ohlc = all(col in result_df.columns for col in ['open', 'high', 'low', 'close'])
        if not has_ohlc:
            self.logger.warning(
                "Відсутні деякі OHLC стовпці. Garman-Klass та інші метрики волатильності не будуть розраховані.")

        # Лічильник доданих ознак
        added_features_count = 0

        # Розрахунок волатильності на основі логарифмічної прибутковості
        log_returns = np.log(result_df[price_column] / result_df[price_column].shift(1))

        # Стандартне відхилення прибутковості для різних вікон
        for window in window_sizes:
            # Класична волатильність як стандартне відхилення прибутковості
            vol_name = f"volatility_{window}d"
            result_df[vol_name] = log_returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(
                252)  # Ануалізована волатильність
            added_features_count += 1

            # Експоненційно зважена волатильність
            ewm_vol_name = f"ewm_volatility_{window}d"
            result_df[ewm_vol_name] = log_returns.ewm(span=window, min_periods=window // 2).std() * np.sqrt(252)
            added_features_count += 1

            # Ковзна сума квадратів прибутковостей (для реалізованої волатильності)
            realized_vol_name = f"realized_volatility_{window}d"
            result_df[realized_vol_name] = np.sqrt(
                np.square(log_returns).rolling(window=window, min_periods=window // 2).sum() * (252 / window))
            added_features_count += 1

            # Відносна волатильність (порівняння з історичною)
            if window > 10:  # Тільки для більших вікон
                long_window = window * 2
                if len(log_returns) > long_window:  # Перевіряємо, що маємо достатньо даних
                    rel_vol_name = f"relative_volatility_{window}d_to_{long_window}d"
                    short_vol = log_returns.rolling(window=window, min_periods=window // 2).std()
                    long_vol = log_returns.rolling(window=long_window, min_periods=long_window // 2).std()
                    result_df[rel_vol_name] = short_vol / long_vol
                    added_features_count += 1

        # Додаткові метрики волатильності, якщо є OHLC дані
        if has_ohlc:
            for window in window_sizes:
                # Garman-Klass волатильність
                gk_vol_name = f"garman_klass_volatility_{window}d"
                result_df['log_hl'] = np.log(result_df['high'] / result_df['low']) ** 2
                result_df['log_co'] = np.log(result_df['close'] / result_df['open']) ** 2

                # Формула Garman-Klass: σ² = 0.5 * log(high/low)² - (2*log(2)-1) * log(close/open)²
                gk_daily = 0.5 * result_df['log_hl'] - (2 * np.log(2) - 1) * result_df['log_co']
                result_df[gk_vol_name] = np.sqrt(gk_daily.rolling(window=window, min_periods=window // 2).mean() * 252)
                added_features_count += 1

                # Yang-Zhang волатильність
                if 'open' in result_df.columns:
                    yz_vol_name = f"yang_zhang_volatility_{window}d"
                    # Overnight volatility
                    result_df['log_oc'] = np.log(result_df['open'] / result_df['close'].shift(1)) ** 2
                    overnight_vol = result_df['log_oc'].rolling(window=window, min_periods=window // 2).mean()

                    # Open-to-Close volatility
                    oc_vol = result_df['log_co'].rolling(window=window, min_periods=window // 2).mean()

                    # Yang-Zhang: використовує overnight та open-to-close волатильність
                    k = 0.34 / (1.34 + (window + 1) / (window - 1))
                    result_df[yz_vol_name] = np.sqrt((overnight_vol + k * oc_vol + (1 - k) * gk_daily.rolling(
                        window=window, min_periods=window // 2).mean()) * 252)
                    added_features_count += 1

            # Видаляємо тимчасові колонки
            result_df.drop(['log_hl', 'log_co'], axis=1, inplace=True, errors='ignore')
            if 'log_oc' in result_df.columns:
                result_df.drop('log_oc', axis=1, inplace=True)

        # Парксінсон волатильність (використовує тільки high і low)
        if all(col in result_df.columns for col in ['high', 'low']):
            for window in window_sizes:
                parkinson_name = f"parkinson_volatility_{window}d"
                # Формула Parkinson: σ² = 1/(4*ln(2)) * log(high/low)²
                parkinson_daily = 1 / (4 * np.log(2)) * np.log(result_df['high'] / result_df['low']) ** 2
                result_df[parkinson_name] = np.sqrt(
                    parkinson_daily.rolling(window=window, min_periods=window // 2).mean() * 252)
                added_features_count += 1

        # Заповнюємо NaN значення
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    self.logger.debug(f"Заповнення NaN значень у стовпці {col}")
                    # Заповнюємо NaN методом forward fill, потім backward fill
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                    # Якщо все ще є NaN, заповнюємо медіаною
                    if result_df[col].isna().any():
                        median_val = result_df[col].median()
                        if pd.isna(median_val):  # Якщо медіана теж NaN
                            result_df[col] = result_df[col].fillna(0)
                        else:
                            result_df[col] = result_df[col].fillna(median_val)

        self.logger.info(f"Додано {added_features_count} ознак волатильності")

        return result_df

    def create_ratio_features(self, data: pd.DataFrame,
                              numerators: List[str],
                              denominators: List[str]) -> pd.DataFrame:

        self.logger.info("Створення ознак-співвідношень...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що всі зазначені стовпці існують в даних
        missing_numerators = [col for col in numerators if col not in result_df.columns]
        missing_denominators = [col for col in denominators if col not in result_df.columns]

        if missing_numerators:
            self.logger.warning(f"Стовпці чисельника {missing_numerators} не знайдено в даних і будуть пропущені")
            numerators = [col for col in numerators if col in result_df.columns]

        if missing_denominators:
            self.logger.warning(f"Стовпці знаменника {missing_denominators} не знайдено в даних і будуть пропущені")
            denominators = [col for col in denominators if col in result_df.columns]

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not numerators or not denominators:
            self.logger.error("Немає доступних стовпців для створення співвідношень")
            return result_df

        # Лічильник доданих ознак
        added_features_count = 0

        # Створюємо всі можливі комбінації співвідношень
        for num_col in numerators:
            for den_col in denominators:
                # Пропускаємо деякі комбінації, якщо чисельник і знаменник однакові
                if num_col == den_col:
                    self.logger.debug(f"Пропускаємо співвідношення {num_col}/{den_col} (однакові стовпці)")
                    continue

                # Створюємо назву нової ознаки
                ratio_name = f"ratio_{num_col}_to_{den_col}"

                # Обробляємо випадки з нульовими знаменниками
                # Використовуємо numpy.divide з параметром where для безпечного ділення
                self.logger.debug(f"Створюємо співвідношення {ratio_name}")

                # Перевіряємо, чи є нульові значення в знаменнику
                zero_denominator_count = (result_df[den_col] == 0).sum()
                if zero_denominator_count > 0:
                    self.logger.warning(f"Знаменник {den_col} містить {zero_denominator_count} нульових значень")

                    # Використовуємо безпечне ділення: ігноруємо ділення на нуль
                    # і встановлюємо спеціальне значення для таких випадків
                    denominator = result_df[den_col].copy()

                    # Створюємо маску для ненульових значень
                    non_zero_mask = (denominator != 0)

                    # Виконуємо ділення тільки для ненульових знаменників
                    result_df[ratio_name] = np.nan  # спочатку встановлюємо NaN
                    result_df.loc[non_zero_mask, ratio_name] = result_df.loc[non_zero_mask, num_col] / denominator[
                        non_zero_mask]

                    # Для нульових знаменників можна встановити спеціальне значення або залишити NaN
                    # Тут ми залишаємо NaN і потім заповнюємо їх
                else:
                    # Якщо нульових знаменників немає, просто ділимо
                    result_df[ratio_name] = result_df[num_col] / result_df[den_col]

                # Обробляємо випадки з нескінченностями (можуть виникнути при діленні на дуже малі числа)
                inf_count = np.isinf(result_df[ratio_name]).sum()
                if inf_count > 0:
                    self.logger.warning(f"Співвідношення {ratio_name} містить {inf_count} нескінченних значень")
                    # Замінюємо нескінченності на NaN для подальшої обробки
                    result_df[ratio_name].replace([np.inf, -np.inf], np.nan, inplace=True)

                # Заповнюємо NaN значення (якщо є)
                if result_df[ratio_name].isna().any():
                    # Заповнюємо NaN медіаною стовпця
                    median_val = result_df[ratio_name].median()
                    if pd.isna(median_val):  # Якщо медіана теж NaN
                        result_df[ratio_name] = result_df[ratio_name].fillna(0)
                        self.logger.debug(f"Заповнення NaN значень у стовпці {ratio_name} нулями")
                    else:
                        result_df[ratio_name] = result_df[ratio_name].fillna(median_val)
                        self.logger.debug(f"Заповнення NaN значень у стовпці {ratio_name} медіаною: {median_val}")

                # Додаємо опціональне обмеження на великі значення
                # Можна використовувати вінсоризацію або кліпінг
                # Тут використовуємо простий кліпінг на основі перцентилів
                q_low, q_high = result_df[ratio_name].quantile([0.01, 0.99])
                result_df[ratio_name] = result_df[ratio_name].clip(q_low, q_high)

                added_features_count += 1

        self.logger.info(f"Додано {added_features_count} ознак-співвідношень")

        return result_df

    def create_crossover_features(self, data: pd.DataFrame,
                                  fast_columns: List[str],
                                  slow_columns: List[str]) -> pd.DataFrame:

        self.logger.info("Створення ознак перетинів індикаторів...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що всі зазначені стовпці існують в даних
        missing_fast = [col for col in fast_columns if col not in result_df.columns]
        missing_slow = [col for col in slow_columns if col not in result_df.columns]

        if missing_fast:
            self.logger.warning(f"Швидкі індикатори {missing_fast} не знайдено в даних і будуть пропущені")
            fast_columns = [col for col in fast_columns if col in result_df.columns]

        if missing_slow:
            self.logger.warning(f"Повільні індикатори {missing_slow} не знайдено в даних і будуть пропущені")
            slow_columns = [col for col in slow_columns if col in result_df.columns]

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not fast_columns or not slow_columns:
            self.logger.error("Немає доступних індикаторів для створення перетинів")
            return result_df

        # Лічильник доданих ознак
        added_features_count = 0

        # Для кожної пари індикаторів створюємо ознаки перетинів
        for fast_col in fast_columns:
            for slow_col in slow_columns:
                # Пропускаємо пари однакових індикаторів
                if fast_col == slow_col:
                    self.logger.debug(f"Пропускаємо пару {fast_col}/{slow_col} (однакові індикатори)")
                    continue

                # Базова назва для ознак цієї пари
                base_name = f"{fast_col}_x_{slow_col}"

                # 1. Створюємо ознаку відносної різниці між індикаторами
                diff_name = f"{base_name}_diff"
                result_df[diff_name] = result_df[fast_col] - result_df[slow_col]
                added_features_count += 1

                # 2. Створюємо ознаку відносної різниці (у відсотках)
                rel_diff_name = f"{base_name}_rel_diff"
                # Уникаємо ділення на нуль
                non_zero_mask = (result_df[slow_col] != 0)
                result_df[rel_diff_name] = np.nan
                result_df.loc[non_zero_mask, rel_diff_name] = (
                        (result_df.loc[non_zero_mask, fast_col] / result_df.loc[non_zero_mask, slow_col] - 1) * 100
                )
                # Заповнюємо NaN значення
                result_df[rel_diff_name].fillna(0, inplace=True)
                added_features_count += 1

                # 3. Створюємо бінарні ознаки перетинів
                # Визначаємо попередні значення різниці для виявлення перетинів
                prev_diff = result_df[diff_name].shift(1)

                # Golden Cross: швидкий індикатор перетинає повільний знизу вгору
                golden_cross_name = f"{base_name}_golden_cross"
                result_df[golden_cross_name] = ((result_df[diff_name] > 0) & (prev_diff <= 0)).astype(int)
                added_features_count += 1

                # Death Cross: швидкий індикатор перетинає повільний згори вниз
                death_cross_name = f"{base_name}_death_cross"
                result_df[death_cross_name] = ((result_df[diff_name] < 0) & (prev_diff >= 0)).astype(int)
                added_features_count += 1

                # 4. Створюємо ознаку тривалості поточного стану (кількість періодів після останнього перетину)
                duration_name = f"{base_name}_state_duration"

                # Ініціалізуємо значення тривалості
                result_df[duration_name] = 0

                # Знаходимо індекси всіх перетинів (обох типів)
                all_crosses = (result_df[golden_cross_name] == 1) | (result_df[death_cross_name] == 1)
                cross_indices = np.where(all_crosses)[0]

                if len(cross_indices) > 0:
                    # Для кожного сегмента між перетинами встановлюємо тривалість
                    prev_idx = 0
                    for idx in cross_indices:
                        if idx > 0:  # Пропускаємо перший перетин (немає даних до нього)
                            # Збільшуємо тривалість для всіх точок у сегменті
                            for i in range(prev_idx, idx):
                                result_df.iloc[i, result_df.columns.get_loc(duration_name)] = i - prev_idx
                        prev_idx = idx

                    # Обробляємо останній сегмент до кінця даних
                    for i in range(prev_idx, len(result_df)):
                        result_df.iloc[i, result_df.columns.get_loc(duration_name)] = i - prev_idx

                added_features_count += 1

                # 5. Додаємо ознаку напрямку (1 якщо швидкий вище повільного, -1 якщо нижче)
                direction_name = f"{base_name}_direction"
                result_df[direction_name] = np.sign(result_df[diff_name]).fillna(0).astype(int)
                added_features_count += 1

                # 6. Додаємо ознаку кутового коефіцієнта (нахилу) між індикаторами
                # Обчислюємо різницю похідних індикаторів для оцінки відносної швидкості зміни
                slope_name = f"{base_name}_slope_diff"
                # Використовуємо різницю за 3 періоди для стабільнішого результату
                fast_slope = result_df[fast_col].diff(3)
                slow_slope = result_df[slow_col].diff(3)
                result_df[slope_name] = fast_slope - slow_slope
                added_features_count += 1

        # Заповнюємо NaN значення для нових ознак
        for col in result_df.columns:
            if col not in data.columns:  # Перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    self.logger.debug(f"Заповнення NaN значень у стовпці {col}")
                    # Для бінарних ознак (перетини) використовуємо 0
                    if col.endswith('_cross'):
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        # Для інших ознак використовуємо прямий і зворотній метод заповнення
                        result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                        # Якщо все ще є NaN, заповнюємо нулями
                        if result_df[col].isna().any():
                            result_df[col] = result_df[col].fillna(0)

        self.logger.info(f"Додано {added_features_count} ознак перетинів індикаторів")

        return result_df

    def create_candle_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Створення ознак на основі патернів свічок...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що необхідні OHLCV колонки існують
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in result_df.columns]

        if missing_columns:
            self.logger.error(f"Відсутні необхідні колонки для патернів свічок: {missing_columns}")
            return result_df

        # Лічильник доданих ознак
        added_features_count = 0

        # --- Базові властивості свічок ---

        # 1. Тіло свічки (абсолютне)
        result_df['candle_body'] = abs(result_df['close'] - result_df['open'])
        added_features_count += 1

        # 2. Верхня тінь
        result_df['upper_shadow'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
        added_features_count += 1

        # 3. Нижня тінь
        result_df['lower_shadow'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
        added_features_count += 1

        # 4. Повний розмах свічки
        result_df['candle_range'] = result_df['high'] - result_df['low']
        added_features_count += 1

        # 5. Відносний розмір тіла (порівняно з повним розмахом)
        # Уникаємо ділення на нуль
        non_zero_range = result_df['candle_range'] != 0
        result_df['rel_body_size'] = np.nan
        result_df.loc[non_zero_range, 'rel_body_size'] = (
                result_df.loc[non_zero_range, 'candle_body'] / result_df.loc[non_zero_range, 'candle_range']
        )
        result_df['rel_body_size'].fillna(0, inplace=True)
        added_features_count += 1

        # 6. Напрямок свічки (1 для бичачої, -1 для ведмежої)
        result_df['candle_direction'] = np.sign(result_df['close'] - result_df['open']).fillna(0).astype(int)
        added_features_count += 1

        # --- Патерни з однієї свічки ---

        # 1. Дожі (тіло менше X% від розмаху)
        doji_threshold = 0.1  # 10% від повного розмаху
        result_df['doji'] = (result_df['rel_body_size'] < doji_threshold).astype(int)
        added_features_count += 1

        # 2. Молот (маленьке тіло, коротка верхня тінь, довга нижня тінь)
        hammer_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                (result_df['lower_shadow'] > 2 * result_df['candle_body']) &  # довга нижня тінь
                (result_df['upper_shadow'] < 0.2 * result_df['lower_shadow'])  # коротка верхня тінь
        )
        result_df['hammer'] = hammer_conditions.astype(int)
        added_features_count += 1

        # 3. Перевернутий молот (маленьке тіло, довга верхня тінь, коротка нижня тінь)
        inv_hammer_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                (result_df['upper_shadow'] > 2 * result_df['candle_body']) &  # довга верхня тінь
                (result_df['lower_shadow'] < 0.2 * result_df['upper_shadow'])  # коротка нижня тінь
        )
        result_df['inverted_hammer'] = inv_hammer_conditions.astype(int)
        added_features_count += 1

        # 4. Довгі свічки (тіло більше X% від середнього тіла за N періодів)
        window = 20
        avg_body = result_df['candle_body'].rolling(window=window).mean()
        result_df['long_candle'] = (result_df['candle_body'] > 1.5 * avg_body).astype(int)
        added_features_count += 1

        # 5. Марібозу (свічка майже без тіней)
        marubozu_threshold = 0.05  # тіні менше 5% від розмаху
        marubozu_conditions = (
                (result_df['upper_shadow'] < marubozu_threshold * result_df['candle_range']) &
                (result_df['lower_shadow'] < marubozu_threshold * result_df['candle_range']) &
                (result_df['rel_body_size'] > 0.9)  # тіло займає більше 90% розмаху
        )
        result_df['marubozu'] = marubozu_conditions.astype(int)
        added_features_count += 1

        # --- Патерни з декількох свічок ---

        # 1. Поглинання (bullish/bearish engulfing)
        # Бичаче поглинання: ведмежа свічка, за якою йде більша бичача
        bullish_engulfing = (
                (result_df['candle_direction'].shift(1) == -1) &  # попередня свічка ведмежа
                (result_df['candle_direction'] == 1) &  # поточна свічка бичача
                (result_df['open'] < result_df['close'].shift(1)) &  # відкриття нижче закриття попередньої
                (result_df['close'] > result_df['open'].shift(1))  # закриття вище відкриття попередньої
        )
        result_df['bullish_engulfing'] = bullish_engulfing.astype(int)
        added_features_count += 1

        # Ведмеже поглинання: бичача свічка, за якою йде більша ведмежа
        bearish_engulfing = (
                (result_df['candle_direction'].shift(1) == 1) &  # попередня свічка бичача
                (result_df['candle_direction'] == -1) &  # поточна свічка ведмежа
                (result_df['open'] > result_df['close'].shift(1)) &  # відкриття вище закриття попередньої
                (result_df['close'] < result_df['open'].shift(1))  # закриття нижче відкриття попередньої
        )
        result_df['bearish_engulfing'] = bearish_engulfing.astype(int)
        added_features_count += 1

        # 2. Ранкова зірка (morning star) - ведмежа свічка, за нею маленька, потім бичача
        morning_star = (
                (result_df['candle_direction'].shift(2) == -1) &  # перша свічка ведмежа
                (result_df['rel_body_size'].shift(1) < 0.3) &  # друга свічка маленька
                (result_df['candle_direction'] == 1) &  # третя свічка бичача
                (result_df['close'] > (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
        # закриття вище середини першої свічки
        )
        result_df['morning_star'] = morning_star.astype(int)
        added_features_count += 1

        # 3. Вечірня зірка (evening star) - бичача свічка, за нею маленька, потім ведмежа
        evening_star = (
                (result_df['candle_direction'].shift(2) == 1) &  # перша свічка бичача
                (result_df['rel_body_size'].shift(1) < 0.3) &  # друга свічка маленька
                (result_df['candle_direction'] == -1) &  # третя свічка ведмежа
                (result_df['close'] < (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
        # закриття нижче середини першої свічки
        )
        result_df['evening_star'] = evening_star.astype(int)
        added_features_count += 1

        # 4. Три білих солдати (three white soldiers) - три послідовні бичачі свічки з закриттям вище попереднього
        three_white_soldiers = (
                (result_df['candle_direction'].shift(2) == 1) &  # перша свічка бичача
                (result_df['candle_direction'].shift(1) == 1) &  # друга свічка бичача
                (result_df['candle_direction'] == 1) &  # третя свічка бичача
                (result_df['close'].shift(1) > result_df['close'].shift(2)) &  # друга закривається вище першої
                (result_df['close'] > result_df['close'].shift(1))  # третя закривається вище другої
        )
        result_df['three_white_soldiers'] = three_white_soldiers.astype(int)
        added_features_count += 1

        # 5. Три чорні ворони (three black crows) - три послідовні ведмежі свічки з закриттям нижче попереднього
        three_black_crows = (
                (result_df['candle_direction'].shift(2) == -1) &  # перша свічка ведмежа
                (result_df['candle_direction'].shift(1) == -1) &  # друга свічка ведмежа
                (result_df['candle_direction'] == -1) &  # третя свічка ведмежа
                (result_df['close'].shift(1) < result_df['close'].shift(2)) &  # друга закривається нижче першої
                (result_df['close'] < result_df['close'].shift(1))  # третя закривається нижче другої
        )
        result_df['three_black_crows'] = three_black_crows.astype(int)
        added_features_count += 1

        # 6. Зірка доджі (doji star) - свічка, за якою йде доджі
        result_df['doji_star'] = (result_df['doji'] & (result_df['doji'].shift(1) == 0)).astype(int)
        added_features_count += 1

        # 7. Пінцет (зверху/знизу) - дві свічки з однаковими максимумами/мінімумами
        pinbar_tolerance = 0.001  # допустима різниця для "однакових" значень

        # Верхній пінцет (однакові максимуми)
        top_pinbar = (
                (abs(result_df['high'] - result_df['high'].shift(1)) < pinbar_tolerance * result_df['high']) &
                (result_df['candle_direction'].shift(1) != result_df['candle_direction'])  # різний напрямок свічок
        )
        result_df['top_pinbar'] = top_pinbar.astype(int)
        added_features_count += 1

        # Нижній пінцет (однакові мінімуми)
        bottom_pinbar = (
                (abs(result_df['low'] - result_df['low'].shift(1)) < pinbar_tolerance * result_df['low']) &
                (result_df['candle_direction'].shift(1) != result_df['candle_direction'])  # різний напрямок свічок
        )
        result_df['bottom_pinbar'] = bottom_pinbar.astype(int)
        added_features_count += 1

        # Заповнюємо пропущені значення (особливо на початку ряду через використання .shift())
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                result_df[col] = result_df[col].fillna(0)  # для бінарних ознак підходить 0

        self.logger.info(f"Додано {added_features_count} ознак на основі патернів свічок")

        return result_df

    def create_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Створення специфічних індикаторів для криптовалют...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо наявність необхідних колонок
        required_columns = {
            'basic': ['close'],
            'volume': ['volume', 'close'],
            'ohlcv': ['open', 'high', 'low', 'close', 'volume']
        }

        # Перевіряємо доступність даних для різних груп індикаторів
        available_groups = []
        missing_columns_info = {}

        for group, cols in required_columns.items():
            missing = [col for col in cols if col not in result_df.columns]
            if not missing:
                available_groups.append(group)
            else:
                missing_columns_info[group] = missing

        if missing_columns_info:
            for group, missing in missing_columns_info.items():
                self.logger.warning(f"Для групи '{group}' відсутні колонки: {missing}")

        if not available_groups:
            self.logger.error("Немає достатньо даних для розрахунку жодного індикатора")
            return result_df

        # Лічильник доданих ознак
        added_features_count = 0

        # --- 1. Chaikin Money Flow (CMF) ---
        if 'ohlcv' in available_groups:
            # Кількість періодів для розрахунку CMF
            for period in [20]:
                # Money Flow Multiplier
                mfm = ((result_df['close'] - result_df['low']) - (result_df['high'] - result_df['close'])) / (
                            result_df['high'] - result_df['low'])
                # Замінюємо нескінченні значення на нуль (коли high=low)
                mfm = mfm.replace([np.inf, -np.inf], 0)

                # Money Flow Volume
                mfv = mfm * result_df['volume']

                # Chaikin Money Flow
                cmf_name = f'cmf_{period}'
                result_df[cmf_name] = mfv.rolling(window=period).sum() / result_df['volume'].rolling(
                    window=period).sum()
                added_features_count += 1

                # Згладжений CMF (EMA на 10 періодів)
                smooth_cmf_name = f'smooth_cmf_{period}'
                result_df[smooth_cmf_name] = result_df[cmf_name].ewm(span=10, min_periods=5).mean()
                added_features_count += 1

        # --- 2. On Balance Volume (OBV) ---
        if 'volume' in available_groups:
            # Розрахунок OBV (класичний алгоритм)
            result_df['price_change'] = result_df['close'].diff()
            result_df['obv'] = 0

            # Перший OBV дорівнює першому обсягу
            if len(result_df) > 0:
                result_df.loc[result_df.index[0], 'obv'] = result_df.loc[result_df.index[0], 'volume']

            # Розрахунок OBV для всіх наступних точок
            for i in range(1, len(result_df)):
                prev_obv = result_df.loc[result_df.index[i - 1], 'obv']
                curr_volume = result_df.loc[result_df.index[i], 'volume']
                price_change = result_df.loc[result_df.index[i], 'price_change']

                if price_change > 0:  # Ціна зросла
                    result_df.loc[result_df.index[i], 'obv'] = prev_obv + curr_volume
                elif price_change < 0:  # Ціна знизилась
                    result_df.loc[result_df.index[i], 'obv'] = prev_obv - curr_volume
                else:  # Ціна не змінилась
                    result_df.loc[result_df.index[i], 'obv'] = prev_obv

            # Видаляємо допоміжний стовпець
            result_df.drop('price_change', axis=1, inplace=True)
            added_features_count += 1

            # OBV EMA Signal - ковзне середнє для OBV
            for period in [20]:
                obv_signal_name = f'obv_signal_{period}'
                result_df[obv_signal_name] = result_df['obv'].ewm(span=period, min_periods=period // 2).mean()
                added_features_count += 1

            # OBV Difference - різниця між OBV та його сигнальною лінією
            result_df['obv_diff'] = result_df['obv'] - result_df['obv_signal_20']
            added_features_count += 1

        # --- 3. Accumulation/Distribution Line (ADL) ---
        if 'ohlcv' in available_groups:
            # Money Flow Multiplier
            mfm = ((result_df['close'] - result_df['low']) - (result_df['high'] - result_df['close'])) / (
                        result_df['high'] - result_df['low'])
            mfm = mfm.replace([np.inf, -np.inf], 0)

            # Money Flow Volume
            mfv = mfm * result_df['volume']

            # ADL - кумулятивна сума Money Flow Volume
            result_df['adl'] = mfv.cumsum()
            added_features_count += 1

            # ADL EMA сигнальна лінія
            for period in [14]:
                adl_signal_name = f'adl_signal_{period}'
                result_df[adl_signal_name] = result_df['adl'].ewm(span=period, min_periods=period // 2).mean()
                added_features_count += 1

        # --- 4. Price Volume Trend (PVT) ---
        if 'volume' in available_groups:
            # Розрахунок процентної зміни ціни
            result_df['price_pct_change'] = result_df['close'].pct_change()

            # PVT = Попередній PVT + (Процентна зміна ціни * Обсяг)
            result_df['pvt'] = result_df['price_pct_change'] * result_df['volume']
            result_df['pvt'] = result_df['pvt'].cumsum()
            added_features_count += 1

            # Сигнальна лінія PVT
            for period in [20]:
                pvt_signal_name = f'pvt_signal_{period}'
                result_df[pvt_signal_name] = result_df['pvt'].ewm(span=period, min_periods=period // 2).mean()
                added_features_count += 1

            # Видаляємо допоміжний стовпець
            result_df.drop('price_pct_change', axis=1, inplace=True)
            # --- 5. Money Flow Index (MFI) ---
            if 'ohlcv' in available_groups:
                for period in [14]:
                    # Типова ціна
                    result_df['typical_price'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3

                    # Raw Money Flow = Типова ціна * Обсяг
                    result_df['raw_money_flow'] = result_df['typical_price'] * result_df['volume']

                    # Позитивний і негативний грошовий потік
                    result_df['positive_flow'] = 0.0
                    result_df['negative_flow'] = 0.0

                    # Визначаємо напрямок потоку на основі зміни типової ціни
                    for i in range(1, len(result_df)):
                        if result_df['typical_price'].iloc[i] > result_df['typical_price'].iloc[i - 1]:
                            result_df['positive_flow'].iloc[i] = result_df['raw_money_flow'].iloc[i]
                        else:
                            result_df['negative_flow'].iloc[i] = result_df['raw_money_flow'].iloc[i]

                    # Підрахунок сум позитивного і негативного потоків за вказаний період
                    positive_money_flow = result_df['positive_flow'].rolling(window=period).sum()
                    negative_money_flow = result_df['negative_flow'].rolling(window=period).sum()

                    # Запобігаємо діленню на нуль
                    negative_money_flow = negative_money_flow.replace(0, 1e-9)

                    # Money Ratio
                    money_ratio = positive_money_flow / negative_money_flow

                    # Money Flow Index
                    mfi_name = f'mfi_{period}'
                    result_df[mfi_name] = 100 - (100 / (1 + money_ratio))
                    added_features_count += 1

                    # Видаляємо допоміжні колонки
                    result_df.drop(['typical_price', 'raw_money_flow', 'positive_flow', 'negative_flow'], axis=1,
                                   inplace=True)

            # --- 6. Volume Price Trend (VPT) ---
            if 'volume' in available_groups:
                # Розрахунок VPT
                result_df['price_change_pct'] = result_df['close'].pct_change()
                result_df['vpt'] = (result_df['price_change_pct'] * result_df['volume']).fillna(0).cumsum()
                added_features_count += 1

                # VPT EMA сигнальна лінія
                for period in [20]:
                    vpt_signal_name = f'vpt_signal_{period}'
                    result_df[vpt_signal_name] = result_df['vpt'].ewm(span=period, min_periods=period // 2).mean()
                    added_features_count += 1

                # Видаляємо допоміжний стовпець
                result_df.drop('price_change_pct', axis=1, inplace=True)

            # --- 7. Volume Weighted Average Price (VWAP) ---
            if 'ohlcv' in available_groups:
                # Типова ціна
                result_df['typical_price'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3

                # Обсяг * Типова ціна
                result_df['vol_tp'] = result_df['volume'] * result_df['typical_price']

                # VWAP за різні періоди
                for period in [14, 30]:
                    vwap_name = f'vwap_{period}'
                    result_df[vwap_name] = result_df['vol_tp'].rolling(window=period).sum() / result_df[
                        'volume'].rolling(window=period).sum()
                    added_features_count += 1

                # Видаляємо допоміжні колонки
                result_df.drop(['typical_price', 'vol_tp'], axis=1, inplace=True)

            # --- 8. Relative Volume (по відношенню до середнього) ---
            if 'volume' in available_groups:
                for period in [20]:
                    avg_volume_name = f'avg_volume_{period}'
                    result_df[avg_volume_name] = result_df['volume'].rolling(window=period).mean()

                    rel_volume_name = f'rel_volume_{period}'
                    result_df[rel_volume_name] = result_df['volume'] / result_df[avg_volume_name]
                    added_features_count += 1

            # --- 9. Bollinger Bands на обсязі ---
            if 'volume' in available_groups:
                for period in [20]:
                    # Середній обсяг
                    vol_ma_name = f'volume_ma_{period}'
                    result_df[vol_ma_name] = result_df['volume'].rolling(window=period).mean()

                    # Стандартне відхилення обсягу
                    vol_std = result_df['volume'].rolling(window=period).std()

                    # Bollinger Bands для обсягу
                    vol_upper_name = f'volume_upper_band_{period}'
                    vol_lower_name = f'volume_lower_band_{period}'
                    result_df[vol_upper_name] = result_df[vol_ma_name] + (2 * vol_std)
                    result_df[vol_lower_name] = result_df[vol_ma_name] - (2 * vol_std)
                    added_features_count += 3

            # --- 10. Price Volume Divergence ---
            if 'volume' in available_groups:
                # Нормалізуємо ціну та обсяг для порівняння
                for period in [20]:
                    # Нормалізовані значення (Z-score)
                    norm_price_name = f'norm_price_{period}'
                    norm_volume_name = f'norm_volume_{period}'

                    price_mean = result_df['close'].rolling(window=period).mean()
                    price_std = result_df['close'].rolling(window=period).std()
                    result_df[norm_price_name] = (result_df['close'] - price_mean) / price_std

                    volume_mean = result_df['volume'].rolling(window=period).mean()
                    volume_std = result_df['volume'].rolling(window=period).std()
                    result_df[norm_volume_name] = (result_df['volume'] - volume_mean) / volume_std

                    # Різниця між нормалізованими значеннями (дивергенція)
                    pv_divergence_name = f'price_volume_divergence_{period}'
                    result_df[pv_divergence_name] = result_df[norm_price_name] - result_df[norm_volume_name]
                    added_features_count += 3

            self.logger.info(f"Створено {added_features_count} додаткових специфічних індикаторів для криптовалют")

            # Замінюємо всі нескінченні значення та NaN на нуль
            result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            result_df.fillna(0, inplace=True)

            return result_df

    def create_volume_features(self, data: pd.DataFrame,
                               window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        self.logger.info("Створення ознак на основі об'єму...")

        # Перевірити, що колонка volume існує
        if 'volume' not in data.columns:
            self.logger.warning("Колонка 'volume' відсутня в даних. Ознаки об'єму не будуть створені.")
            return data

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Лічильник доданих ознак
        added_features_count = 0

        # Створити ознаки відносного об'єму
        for window in window_sizes:
            # Ковзне середнє об'єму
            vol_ma_col = f'volume_ma_{window}'
            result_df[vol_ma_col] = result_df['volume'].rolling(window=window, min_periods=1).mean()
            added_features_count += 1

            # Відносний об'єм (поточний об'єм / середній об'єм)
            rel_vol_col = f'rel_volume_{window}'
            result_df[rel_vol_col] = result_df['volume'] / result_df[vol_ma_col]
            added_features_count += 1

            # Стандартне відхилення об'єму
            vol_std_col = f'volume_std_{window}'
            result_df[vol_std_col] = result_df['volume'].rolling(window=window, min_periods=1).std()
            added_features_count += 1

            # Z-score об'єму (наскільки поточний об'єм відхиляється від середнього в одиницях стандартного відхилення)
            vol_zscore_col = f'volume_zscore_{window}'
            result_df[vol_zscore_col] = (result_df['volume'] - result_df[vol_ma_col]) / result_df[vol_std_col]
            # Замінюємо inf значення (які можуть виникнути, якщо std=0)
            result_df[vol_zscore_col].replace([np.inf, -np.inf], 0, inplace=True)
            added_features_count += 1

            # Експоненціальне ковзне середнє об'єму (реагує швидше на зміни)
            vol_ema_col = f'volume_ema_{window}'
            result_df[vol_ema_col] = result_df['volume'].ewm(span=window, min_periods=1).mean()
            added_features_count += 1

            # Зміна об'єму (процентна зміна ковзного середнього)
            vol_change_col = f'volume_change_{window}'
            result_df[vol_change_col] = result_df[vol_ma_col].pct_change(periods=1) * 100
            added_features_count += 1

        # Абсолютна зміна об'єму за 1 період
        result_df['volume_diff_1'] = result_df['volume'].diff(periods=1)
        added_features_count += 1

        # Процентна зміна об'єму за 1 період
        result_df['volume_pct_change_1'] = result_df['volume'].pct_change(periods=1) * 100
        added_features_count += 1

        # Обчислення кумулятивного об'єму за день (якщо дані містять внутрішньоденну інформацію)
        if isinstance(result_df.index, pd.DatetimeIndex):
            result_df['cumulative_daily_volume'] = result_df.groupby(result_df.index.date)['volume'].cumsum()
            added_features_count += 1

        # Індикатор аномального об'єму (об'єм, що перевищує середній у 2+ рази)
        result_df['volume_anomaly'] = (result_df['rel_volume_20'] > 2.0).astype(int)
        added_features_count += 1

        # Замінюємо NaN значення нулями
        result_df.fillna(0, inplace=True)

        self.logger.info(f"Створено {added_features_count} ознак на основі об'єму.")
        return result_df

    def create_datetime_features(self, data: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:

        self.logger.info("Створення ознак на основі дати і часу...")

        # Перевірити, що індекс є DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є DatetimeIndex. Часові ознаки не будуть створені.")
            return data

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Лічильник доданих ознак
        added_features_count = 0

        # Створити базові ознаки
        # Година дня (0-23)
        result_df['hour'] = result_df.index.hour
        added_features_count += 1

        # День тижня (0-6, де 0 - понеділок, 6 - неділя)
        result_df['day_of_week'] = result_df.index.dayofweek
        added_features_count += 1

        # Номер дня місяця (1-31)
        result_df['day_of_month'] = result_df.index.day
        added_features_count += 1

        # Номер місяця (1-12)
        result_df['month'] = result_df.index.month
        added_features_count += 1

        # Номер кварталу (1-4)
        result_df['quarter'] = result_df.index.quarter
        added_features_count += 1

        # Номер року
        result_df['year'] = result_df.index.year
        added_features_count += 1

        # День року (1-366)
        result_df['day_of_year'] = result_df.index.dayofyear
        added_features_count += 1

        # Час доби (ранок, день, вечір, ніч)
        result_df['time_of_day'] = pd.cut(
            result_df.index.hour,
            bins=[-1, 5, 11, 17, 23],
            labels=['night', 'morning', 'day', 'evening']
        ).astype(str)
        added_features_count += 1

        # Чи це вихідний день (субота або неділя)
        result_df['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
        added_features_count += 1

        # Чи це останній день місяця
        result_df['is_month_end'] = result_df.index.is_month_end.astype(int)
        added_features_count += 1

        # Чи це останній день кварталу
        result_df['is_quarter_end'] = result_df.index.is_quarter_end.astype(int)
        added_features_count += 1

        # Чи це останній день року
        result_df['is_year_end'] = result_df.index.is_year_end.astype(int)
        added_features_count += 1

        # Якщо потрібні циклічні ознаки
        if cyclical:
            # Функція для створення циклічних ознак
            def create_cyclical_features(df, col, max_val):
                # Перетворення у радіани
                df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
                df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
                return df, 2  # Повертаємо df і кількість доданих ознак

            # Година (0-23)
            result_df, count = create_cyclical_features(result_df, 'hour', 24)
            added_features_count += count

            # День тижня (0-6)
            result_df, count = create_cyclical_features(result_df, 'day_of_week', 7)
            added_features_count += count

            # День місяця (1-31)
            result_df, count = create_cyclical_features(result_df, 'day_of_month', 31)
            added_features_count += count

            # Місяць (1-12)
            result_df, count = create_cyclical_features(result_df, 'month', 12)
            added_features_count += count

            # День року (1-366)
            result_df, count = create_cyclical_features(result_df, 'day_of_year', 366)
            added_features_count += count

        # Додаткові ознаки для криптовалютного ринку

        # Час доби за UTC (важливо для глобальних криптовалютних ринків)
        if 'hour' in result_df.columns:
            # Сегменти дня за UTC для виявлення патернів активності на різних ринках
            result_df['utc_segment'] = pd.cut(
                result_df['hour'],
                bins=[-1, 2, 8, 14, 20, 23],
                labels=['asia_late', 'asia_main', 'europe_main', 'america_main', 'asia_early']
            ).astype(str)
            added_features_count += 1

        # Опціонально: додавання відміток часу в Unix форматі (для розрахунків темпоральних відмінностей)
        result_df['timestamp'] = result_df.index.astype(np.int64) // 10 ** 9
        added_features_count += 1

        self.logger.info(f"Створено {added_features_count} часових ознак.")
        return result_df

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