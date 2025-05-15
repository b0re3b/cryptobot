import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from arch import arch_model
import concurrent.futures
from functools import lru_cache
from data.db import DatabaseManager
from DMP.DataCleaner import DataCleaner
from models.time_series import TimeSeriesModels
from DMP.AnomalyDetector import AnomalyDetector
from data_collection.feature_engineering import FeatureEngineering
from utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityAnalysis:

    def __init__(self, use_parallel=True, max_workers=4, cache_size=128):
        self.db_manager = DatabaseManager()
        self.volatility_models = {}
        self.regime_models = {}
        self.data_cleaner = DataCleaner()
        self.anomaly_detector = AnomalyDetector(logger=logger)
        self.feature_engineer = FeatureEngineering()
        self.time_series = TimeSeriesModels()

        # Оптимізація: налаштування паралельних обчислень
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.cache_size = cache_size

    @lru_cache(maxsize=128)
    def _calc_log_returns(self, prices_tuple):
        """Хелпер для обчислення логарифмічних прибутків з кешуванням"""
        prices = np.array(prices_tuple)
        return np.log(prices[1:] / prices[:-1])

    def _parallel_process(self, func, items, *args, **kwargs):
        """Виконує обчислення паралельно для списку елементів"""
        if not self.use_parallel or len(items) < 3:
            return {item: func(item, *args, **kwargs) for item in items}

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(func, item, *args, **kwargs): item for item in items}
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    results[item] = future.result()
                except Exception as e:
                    logger.error(f"Помилка при паралельній обробці для {item}: {e}")
                    results[item] = None
        return results

    def get_market_phases(self, volatility_data, lookback_window=90, n_regimes=4):
        try:
            logger.info(f"Визначення фаз ринку за допомогою {n_regimes} режимів")

            # Отримати останні дані в межах вікна аналізу (векторизовано)
            recent_data = volatility_data.iloc[-lookback_window:] if len(
                volatility_data) > lookback_window else volatility_data

            # Розрахувати показники волатильності для всього ринку (векторизовано)
            market_vol = recent_data.mean(axis=1)  # Середня волатильність по всіх активах
            vol_dispersion = recent_data.std(axis=1)  # Розсіювання (дисперсія) волатильності
            vol_trend = market_vol.diff(5).rolling(window=10).mean()  # Тренд волатильності (різниця + згладжування)

            # Об'єднати показники у матрицю ознак для кластеризації
            features = pd.DataFrame({
                'market_vol': market_vol,
                'vol_dispersion': vol_dispersion,
                'vol_trend': vol_trend
            }).dropna()

            # Якщо даних недостатньо — повертаємо порожню серію
            if len(features) < 10:
                logger.warning("Недостатньо даних для визначення фаз ринку")
                return pd.Series(index=volatility_data.index)

            # Нормалізуємо ознаки для кластеризації
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Застосовуємо кластеризацію KMeans для виділення фаз ринку
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)  # n_init явно вказано для сумісності
            phases = kmeans.fit_predict(scaled_features)

            # Отримати центроїди кластерів
            centroids = kmeans.cluster_centers_

            # Визначаємо порядок фаз за рівнем волатильності (перша ознака)
            vol_level_order = np.argsort(centroids[:, 0])

            # Формуємо список назв для фаз
            phase_names = ['Low Vol', 'Normal Vol', 'High Vol', 'Extreme Vol']
            if n_regimes < 4:
                phase_names = phase_names[:n_regimes]
            elif n_regimes > 4:
                phase_names.extend([f'Корист. фаза {i + 1}' for i in range(n_regimes - 4)])

            # Створюємо відповідність між кластерами та назвами фаз
            phase_mapping = {vol_level_order[i]: phase_names[i] for i in range(n_regimes)}

            # Присвоюємо фази кожному рядку
            phase_series = pd.Series(phases, index=features.index).map(phase_mapping)

            # Розширюємо серію на повний період вхідних даних, заповнюючи пропуски
            full_phase_series = pd.Series(index=volatility_data.index)
            full_phase_series.loc[phase_series.index] = phase_series
            full_phase_series = full_phase_series.ffill().bfill()  # Заповнення в обидва боки

            # Зберігаємо модель для подальшого використання
            self.regime_models[f"market_phases_{n_regimes}"] = {
                'model': kmeans,
                'scaler': scaler,
                'features': list(features.columns),
                'mapping': phase_mapping
            }

            logger.info(f"Успішно визначено {n_regimes} фази ринку")
            return full_phase_series

        except Exception as e:
            logger.error(f"Помилка при визначенні фаз ринку: {e}")
            return pd.Series(index=volatility_data.index)

    def calculate_historical_volatility(self, price_data, window=14, trading_periods=365, annualize=True):
        """Розраховує історичну волатильність (оптимізовано векторизацією)"""
        # Обробка вхідних даних
        price_data = pd.Series(price_data) if not isinstance(price_data, pd.Series) else price_data

        # Векторизовані обчислення
        log_returns = np.log(price_data / price_data.shift(1)).dropna()

        # Оптимізоване обчислення волатильності
        rolling_vol = log_returns.rolling(window=window).std()

        # Анулізація за потреби
        if annualize:
            rolling_vol = rolling_vol * np.sqrt(trading_periods)

        return rolling_vol

    def calculate_parkinson_volatility(self, ohlc_data, window=14, trading_periods=365):
        """Розраховує волатильність Паркінсона (оптимізовано)"""
        # Перевірка, чи є ohlc_data DataFrame
        if not isinstance(ohlc_data, pd.DataFrame):
            logger.error("Дані OHLC повинні бути DataFrame")
            return pd.Series()

        try:
            # Векторизований розрахунок нормалізованого діапазону high-low
            hl_range = np.log(ohlc_data['high'] / ohlc_data['low'])
            parkinson = pd.Series(hl_range ** 2 / (4 * np.log(2)), index=ohlc_data.index)

            # Обчислення волатильності та анулізація
            rolling_parkinson = np.sqrt(parkinson.rolling(window=window).mean() * trading_periods)

            return rolling_parkinson
        except KeyError:
            logger.error("Дані OHLC повинні містити стовпці 'high' і 'low'")
            return pd.Series()

    def calculate_garman_klass_volatility(self, ohlc_data, window=14, trading_periods=365):
        """Розраховує волатильність Гарман-Класс (оптимізовано)"""
        try:
            # Векторизоване обчислення компонентів
            log_hl = np.log(ohlc_data['high'] / ohlc_data['low']) ** 2 * 0.5
            log_co = np.log(ohlc_data['close'] / ohlc_data['open']) ** 2 * (2 * np.log(2) - 1)

            # Комбінування компонентів
            gk = pd.Series(log_hl - log_co, index=ohlc_data.index)

            # Обчислення волатильності та анулізація
            rolling_gk = np.sqrt(gk.rolling(window=window).mean() * trading_periods)

            return rolling_gk
        except KeyError:
            logger.error("Дані OHLC повинні містити стовпці 'high', 'low', 'close' і 'open'")
            return pd.Series()

    def calculate_yang_zhang_volatility(self, ohlc_data, window=14, trading_periods=365):
        """Розраховує волатильність Янг Чжанг (оптимізовано)"""
        try:
            # Оптимізація: обчислюємо всі компоненти одночасно
            # Нічна волатильність (close to open)
            overnight_returns = np.log(ohlc_data['open'] / ohlc_data['close'].shift(1))
            overnight_vol = overnight_returns.rolling(window=window).var()

            # Волатильність open-close
            open_close_returns = np.log(ohlc_data['close'] / ohlc_data['open'])
            open_close_vol = open_close_returns.rolling(window=window).var()

            # Волатильність Rogers-Satchell
            log_ho = np.log(ohlc_data['high'] / ohlc_data['open'])
            log_lo = np.log(ohlc_data['low'] / ohlc_data['open'])
            log_hc = np.log(ohlc_data['high'] / ohlc_data['close'])
            log_lc = np.log(ohlc_data['low'] / ohlc_data['close'])

            rs_vol = log_ho * (log_ho - log_lo) + log_lc * (log_lc - log_hc)
            rs_vol = rs_vol.rolling(window=window).mean()

            # Обчислення волатильності Yang-Zhang з k=0.34 (рекомендоване значення)
            k = 0.34
            yang_zhang = overnight_vol + k * open_close_vol + (1 - k) * rs_vol

            # Анулізація
            yang_zhang = np.sqrt(yang_zhang * trading_periods)

            return yang_zhang
        except KeyError:
            logger.error("Дані OHLC повинні містити стовпці 'high', 'low', 'close' і 'open'")
            return pd.Series()

    # Оптимізація: кешування результатів GARCH моделей
    @lru_cache(maxsize=32)
    def fit_garch_model(self, returns_key, p=1, q=1, model_type='GARCH'):
        """
        Підгонка GARCH моделі з кешуванням для покращення продуктивності
        returns_key: кортеж для кешування (хеш-значення серії повернення)
        """
        try:
            # Перетворюємо ключ назад на серію
            returns = pd.Series(returns_key)

            logger.info(f"Підгонка {model_type}({p},{q}) моделі")

            # Очистка та підготовка даних
            clean_returns = self.data_cleaner.clean_data(returns)

            # Налаштування моделі
            if model_type == 'EGARCH':
                model = arch_model(clean_returns, vol='EGARCH', p=p, q=q)
            elif model_type == 'GJR-GARCH':
                model = arch_model(clean_returns, vol='GARCH', p=p, o=1, q=q)
            else:  # Default to GARCH
                model = arch_model(clean_returns, vol='GARCH', p=p, q=q)

            # Підгонка моделі з оптимізацією
            fitted_model = model.fit(disp='off', options={'maxiter': 500})

            # Отримання прогнозу
            forecast = fitted_model.forecast(horizon=30)

            # Зберігання моделі для подальшого використання
            self.volatility_models[f"{model_type}_{p}_{q}"] = fitted_model

            logger.info(f"Модель {model_type}({p},{q}) успішно підігнана")
            return fitted_model, forecast

        except Exception as e:
            logger.error(f"Помилка підгонки GARCH моделі: {e}")
            return None, None

    def detect_volatility_regimes(self, volatility_series, n_regimes=3, method='kmeans'):
        """Виявлення режимів волатильності (оптимізовано)"""
        try:
            logger.info(f"Виявлення {n_regimes} режимів волатильності за методом {method}")

            # Очистка даних та перетворення для кластеризації
            clean_vol = volatility_series.dropna().values.reshape(-1, 1)

            if method == 'kmeans':
                # Оптимізований KMeans з більш ефективною ініціалізацією
                kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10, max_iter=300)
                regimes = kmeans.fit_predict(clean_vol)

                # Створення серії з оригінальним індексом
                regime_series = pd.Series(regimes, index=volatility_series.dropna().index)

                # Призначення осмислених міток (0 = low, 1 = medium, 2 = high)
                # Сортування кластерів за їх центроїдами
                centroids = kmeans.cluster_centers_.flatten()
                centroid_mapping = {i: rank for rank, i in enumerate(np.argsort(centroids))}
                regime_series = regime_series.map(centroid_mapping)

            elif method == 'threshold':
                # Оптимізація: використання NumPy для швидких обчислень
                # Використання порогів процентилів
                values = volatility_series.dropna().values
                thresholds = [np.percentile(values, q * 100) for q in np.linspace(0, 1, n_regimes + 1)[1:-1]]

                # Ініціалізація з найнижчим режимом
                regimes = np.zeros(len(values))

                # Призначення режимів на основі порогів
                for i, threshold in enumerate(thresholds, 1):
                    regimes[values > threshold] = i

                regime_series = pd.Series(regimes, index=volatility_series.dropna().index)

            else:
                # Оптимізований стандартний метод
                regime_series = pd.qcut(volatility_series.dropna(), n_regimes, labels=False)

            # Зберігання моделі режиму
            self.regime_models[f"{method}_{n_regimes}"] = {
                'model': kmeans if method == 'kmeans' else None,
                'thresholds': thresholds if method == 'threshold' else None
            }

            logger.info(f"Успішно виявлено {n_regimes} режимів волатильності")
            return regime_series

        except Exception as e:
            logger.error(f"Помилка при виявленні режимів волатильності: {e}")
            return None

    def analyze_volatility_clustering(self, returns, max_lag=30):
        """Аналіз кластеризації волатильності (оптимізовано)"""
        try:
            # Обчислення квадратів прибутку
            squared_returns = returns ** 2

            # Обчислення автокореляції
            # Оптимізація: обробка пропусків перед обчисленням ACF
            clean_squared_returns = squared_returns.dropna()

            if len(clean_squared_returns) < max_lag + 1:
                logger.warning(f"Недостатньо даних для обчислення автокореляції з {max_lag} лагами")
                return pd.DataFrame({'lag': range(len(clean_squared_returns)), 'autocorrelation': 0})

            acf_values = acf(clean_squared_returns, nlags=max_lag, fft=True)  # FFT для швидкої обробки

            # Створення результуючого DataFrame
            acf_df = pd.DataFrame({
                'lag': range(len(acf_values)),
                'autocorrelation': acf_values
            })

            return acf_df
        except Exception as e:
            logger.error(f"Помилка при аналізі кластеризації волатильності: {e}")
            return pd.DataFrame({'lag': [0], 'autocorrelation': [0]})

    def calculate_volatility_risk_metrics(self, returns, volatility):
        """Обчислення метрик ризику волатильності (оптимізовано для великих наборів даних)"""
        try:
            logger.info("Обчислення метрик ризику волатильності")

            # Видалення пропусків для точних обчислень
            clean_returns = returns.dropna()
            clean_volatility = volatility.reindex(clean_returns.index).dropna()

            # Оптимізація: обчислення VaR за допомогою квантилів NumPy
            var_95 = np.percentile(clean_returns, 5)
            var_99 = np.percentile(clean_returns, 1)

            # Оптимізація: обчислення умовного VaR (Expected Shortfall)
            cvar_95 = clean_returns[clean_returns <= var_95].mean()
            cvar_99 = clean_returns[clean_returns <= var_99].mean()

            # Обчислення волатильності волатильності
            vol_of_vol = clean_volatility.rolling(window=30).std().mean()

            # Обчислення інших метрик ризику
            sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(365)  # Annualized Sharpe

            # Оптимізація: обчислення максимального просідання
            cumulative_returns = clean_returns.cumsum()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = drawdown.min()

            # Повернення метрик як словника
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'vol_of_vol_mean': vol_of_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            }

        except Exception as e:
            logger.error(f"Помилка при обчисленні метрик ризику волатильності: {e}")
            return {
                'var_95': None,
                'var_99': None,
                'cvar_95': None,
                'cvar_99': None,
                'vol_of_vol_mean': None,
                'sharpe_ratio': None,
                'max_drawdown': None
            }

    def compare_volatility_metrics(self, ohlc_data, windows=[14, 30, 60]):
        """Порівняння метрик волатильності (оптимізовано паралельними обчисленнями)"""
        try:
            logger.info(f"Порівняння метрик волатильності для вікон {windows}")

            result = pd.DataFrame(index=ohlc_data.index)

            # Обчислення прибутків
            ohlc_data['returns'] = ohlc_data['close'].pct_change()

            # Оптимізація: паралельне обчислення для різних вікон у великих наборах даних
            if self.use_parallel and len(windows) > 2:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Обчислення історичної волатильності
                    hist_vol_futures = {
                        executor.submit(self.calculate_historical_volatility,
                                        ohlc_data['close'], window=window): f'historical_{window}d'
                        for window in windows
                    }

                    # Обчислення волатильності Паркінсона
                    park_vol_futures = {
                        executor.submit(self.calculate_parkinson_volatility,
                                        ohlc_data, window=window): f'parkinson_{window}d'
                        for window in windows
                    }

                    # Обчислення волатильності Гарман-Класс
                    gk_vol_futures = {
                        executor.submit(self.calculate_garman_klass_volatility,
                                        ohlc_data, window=window): f'gk_{window}d'
                        for window in windows
                    }

                    # Обчислення волатильності Янг Чжанг
                    yz_vol_futures = {
                        executor.submit(self.calculate_yang_zhang_volatility,
                                        ohlc_data, window=window): f'yz_{window}d'
                        for window in windows
                    }

                    # Збір результатів
                    for future_dict in [hist_vol_futures, park_vol_futures, gk_vol_futures, yz_vol_futures]:
                        for future, col_name in future_dict.items():
                            try:
                                result[col_name] = future.result()
                            except Exception as e:
                                logger.error(f"Помилка при обчисленні {col_name}: {e}")
            else:
                # Послідовне обчислення для менших наборів даних
                for window in windows:
                    # Обчислення різних метрик волатильності
                    result[f'historical_{window}d'] = self.calculate_historical_volatility(
                        ohlc_data['close'], window=window)

                    result[f'parkinson_{window}d'] = self.calculate_parkinson_volatility(
                        ohlc_data, window=window)

                    result[f'gk_{window}d'] = self.calculate_garman_klass_volatility(
                        ohlc_data, window=window)

                    result[f'yz_{window}d'] = self.calculate_yang_zhang_volatility(
                        ohlc_data, window=window)

            logger.info("Успішно порівняно метрики волатильності")
            return result

        except Exception as e:
            logger.error(f"Помилка при порівнянні метрик волатильності: {e}")
            return pd.DataFrame(index=ohlc_data.index)

    def identify_volatility_breakouts(self, volatility_series, window=20, std_dev=2):
        """Виявлення проривів волатильності (оптимізовано)"""
        try:
            # Обчислення ковзного середнього та стандартного відхилення
            rolling_mean = volatility_series.rolling(window=window).mean()
            rolling_std = volatility_series.rolling(window=window).std()

            # Обчислення верхнього порогу
            upper_threshold = rolling_mean + std_dev * rolling_std

            # Виявлення проривів
            breakouts = volatility_series > upper_threshold

            logger.info(f"Виявлено {breakouts.sum()} проривів волатильності з {len(breakouts)} точок")
            return breakouts

        except Exception as e:
            logger.error(f"Помилка при виявленні проривів волатильності: {e}")
            return pd.Series(False, index=volatility_series.index)

    def analyze_cross_asset_volatility(self, asset_dict, window=14):
        """Аналіз волатильності між активами (оптимізовано для великої кількості активів)"""
        try:
            logger.info(f"Аналіз кросс-активної волатильності для {len(asset_dict)} активів")

            volatility_dict = {}

            # Оптимізація: паралельне обчислення волатильності для кожного активу
            if self.use_parallel and len(asset_dict) > 3:
                volatility_dict = self._parallel_process(
                    lambda asset_name, prices: (asset_name,
                                                self.calculate_historical_volatility(prices, window=window)),
                    asset_dict.keys(),
                    prices_list=list(asset_dict.values())
                )
            else:
                # Обчислення волатильності для кожного активу послідовно
                for asset_name, price_series in asset_dict.items():
                    volatility_dict[asset_name] = self.calculate_historical_volatility(
                        price_series, window=window)

            # Створення DataFrame з усіма волатильностями
            vol_df = pd.DataFrame(volatility_dict)

            # Обчислення кореляційної матриці
            corr_matrix = vol_df.corr()

            logger.info("Успішно проаналізовано кросс-активну волатильність")
            return corr_matrix

        except Exception as e:
            logger.error(f"Помилка при аналізі кросс-активної волатильності: {e}")
            return pd.DataFrame()

    def extract_seasonality_in_volatility(self, volatility_series, period=7):
        """Витяг сезонності з волатильності (оптимізовано)"""
        try:
            # Перетворення індексу на datetime, якщо це ще не зроблено
            if not isinstance(volatility_series.index, pd.DatetimeIndex):
                volatility_series.index = pd.to_datetime(volatility_series.index)

            if period == 7:
                # Витяг сезонності дня тижня
                day_of_week = volatility_series.groupby(volatility_series.index.dayofweek).mean()
                day_of_week.index = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'П\'ятниця', 'Субота', 'Неділя']
                return day_of_week

            elif period == 24:
                # Витяг сезонності години дня (для внутрішньоденних даних)
                hour_of_day = volatility_series.groupby(volatility_series.index.hour).mean()
                return hour_of_day

            elif period == 30:
                # Витяг сезонності дня місяця
                day_of_month = volatility_series.groupby(volatility_series.index.day).mean()
                return day_of_month

            elif period == 12:
                # Витяг сезонності місяця року
                month_of_year = volatility_series.groupby(volatility_series.index.month).mean()
                month_of_year.index = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер',
                                       'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']
                return month_of_year

        except Exception as e:
            logger.error(f"Помилка при вилученні сезонності з волатильності: {e}")
            return pd.Series()

    def analyze_volatility_term_structure(self, symbol, timeframes=['1h', '4h', '1d', '1w']):
        """Аналіз терміну структури волатильності (оптимізовано)"""
        try:
            logger.info(f"Аналіз терміну структури волатильності для {symbol} на {timeframes}")
            results = {}

            # Оптимізація: паралельний збір даних для різних часових проміжків
            if self.use_parallel and len(timeframes) > 2:
                def process_timeframe(tf):
                    try:
                        data = self.db_manager.get_klines(symbol, timeframe=tf)
                        if data is None or data.empty:
                            return tf, None
                        vol = self.calculate_historical_volatility(data['close'])
                        return tf, {
                            'mean_vol': vol.mean(),
                            'median_vol': vol.median(),
                            'max_vol': vol.max(),
                            'min_vol': vol.min(),
                            'std_vol': vol.std()
                        }
                    except Exception as e:
                        logger.error(f"Помилка при обробці таймфрейму {tf}: {e}")
                        return tf, None

                # Паралельне виконання для кожного часового проміжку
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_tf = {executor.submit(process_timeframe, tf): tf for tf in timeframes}
                    for future in concurrent.futures.as_completed(future_to_tf):
                        tf, result = future.result()
                        if result is not None:
                            results[tf] = result
            else:
                # Послідовна обробка для меншої кількості часових проміжків
                for timeframe in timeframes:
                    data = self.db_manager.get_klines(symbol, timeframe=timeframe)
                    if data is None or data.empty:
                        continue
                    vol = self.calculate_historical_volatility(data['close'])
                    results[timeframe] = {
                        'mean_vol': vol.mean(),
                        'median_vol': vol.median(),
                        'max_vol': vol.max(),
                        'min_vol': vol.min(),
                        'std_vol': vol.std()
                    }

            # Перетворення на DataFrame для зручного аналізу
            term_structure = pd.DataFrame(results).T

            # Додаємо відносну волатильність (нормалізовану до денної)
            if '1d' in term_structure.index and 'mean_vol' in term_structure.columns:
                base_vol = term_structure.loc['1d', 'mean_vol']
                if base_vol > 0:
                    term_structure['rel_to_daily'] = term_structure['mean_vol'] / base_vol

            logger.info(f"Успішно проаналізовано термін структури волатильності для {symbol}")
            return term_structure

        except Exception as e:
            logger.error(f"Помилка при аналізі терміну структури волатильності: {e}")
            return pd.DataFrame()

    def volatility_impulse_response(self, returns, shock_size=3, days=30):
        """Аналіз імпульсного відгуку волатильності (оптимізовано)"""
        try:
            logger.info(f"Обчислення імпульсного відгуку волатильності з розміром шоку {shock_size}")

            # Перетворення вхідних даних на серію, якщо це не серія
            returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns

            # Оптимізація: перетворення в кортеж для кешування (використовуємо хеш-значення)
            returns_key = tuple(returns.values)

            # Отримання базової моделі GARCH
            model, _ = self.fit_garch_model(returns_key, p=1, q=1, model_type='GARCH')

            if model is None:
                logger.warning("Не вдалося підігнати GARCH модель для імпульсного відгуку")
                return None

            # Створення базового прогнозу
            baseline_forecast = model.forecast(horizon=days)
            baseline = baseline_forecast.variance.values[-1]

            # Додавання шоку до останнього повернення
            shocked_returns = returns.copy()
            shock_value = shock_size * shocked_returns.std()
            shocked_returns.iloc[-1] = shock_value

            # Оптимізація: перевикористовуємо параметри моделі замість повторної оцінки з нуля
            model_config = model.params
            shocked_model = arch_model(shocked_returns, vol='GARCH', p=1, q=1)

            # Використання початкових параметрів з базової моделі для швидшої збіжності
            shocked_fit = shocked_model.fit(disp='off', starting_values=model_config)

            # Прогноз після шоку
            shocked_forecast = shocked_fit.forecast(horizon=days)
            shocked_values = shocked_forecast.variance.values[-1]

            # Обчислення імпульсного відгуку (різниця від базового прогнозу)
            impulse = pd.Series(shocked_values - baseline, index=range(1, days + 1))

            # Додатково обчислюємо час напіврозпаду шоку
            half_life = None
            if impulse[0] > 0:
                for i, val in enumerate(impulse):
                    if val <= impulse[0] / 2:
                        half_life = i + 1
                        break

            logger.info(f"Успішно обчислено імпульсний відгук волатильності, час напіврозпаду: {half_life}")

            # Повертаємо імпульсний відгук з додатковими метаданими
            return {
                'impulse': impulse,
                'half_life': half_life,
                'shock_size': shock_size,
                'max_effect': impulse.max(),
                'decay_rate': impulse.diff().mean() if len(impulse) > 1 else 0
            }

        except Exception as e:
            logger.error(f"Помилка при обчисленні імпульсного відгуку волатильності: {e}")
            return None

    def prepare_volatility_features_for_ml(self, ohlc_data, window_sizes=[7, 14, 30], include_regimes=True):
        """Підготовка функцій волатильності для машинного навчання (оптимізовано)"""
        try:
            logger.info(f"Підготовка функцій волатильності для ML з вікнами {window_sizes}")

            # Ініціалізація результуючого DataFrame
            features = pd.DataFrame(index=ohlc_data.index)

            # Обчислення прибутків
            returns = ohlc_data['close'].pct_change()
            features['returns'] = returns

            # Використання модуля feature engineering для створення стандартних функцій волатильності
            # Оптимізація: кешування результатів feature_engineer для повторного використання
            cache_key = hash(tuple(ohlc_data.values.flatten()[:100]))  # Використовуємо перші 100 значень як ключ кешу
            if hasattr(self, '_feature_cache') and cache_key in self._feature_cache:
                vol_features = self._feature_cache[cache_key]
            else:
                vol_features = self.feature_engineer.create_volatility_features(ohlc_data)
                if not hasattr(self, '_feature_cache'):
                    self._feature_cache = {}
                self._feature_cache[cache_key] = vol_features

            features = pd.concat([features, vol_features], axis=1)

            # Оптимізація: паралельне обчислення різних вікон волатильності
            if self.use_parallel and len(window_sizes) > 2:
                def compute_window_features(window):
                    try:
                        window_features = {}
                        # Додавання історичної волатильності
                        hist_vol = self.calculate_historical_volatility(ohlc_data['close'], window=window)
                        window_features[f'hist_vol_{window}d'] = hist_vol

                        # Додавання волатильності Паркінсона
                        park_vol = self.calculate_parkinson_volatility(ohlc_data, window=window)
                        window_features[f'park_vol_{window}d'] = park_vol

                        # Додавання відносної волатильності
                        moving_avg_vol = hist_vol.rolling(window=window * 2).mean()
                        rel_vol = hist_vol / moving_avg_vol
                        window_features[f'rel_vol_{window}d'] = rel_vol

                        # Додавання волатильності волатильності
                        vol_of_vol = hist_vol.rolling(window=window).std()
                        window_features[f'vol_of_vol_{window}d'] = vol_of_vol

                        # Тренд волатильності
                        vol_trend = hist_vol.diff(window)
                        window_features[f'vol_trend_{window}d'] = vol_trend

                        # Відношення діапазону High-Low до волатильності
                        hl_range = (ohlc_data['high'] - ohlc_data['low']) / ohlc_data['close']
                        window_features[f'hl_range_to_vol_{window}d'] = hl_range / hist_vol

                        return window_features
                    except Exception as e:
                        logger.error(f"Помилка при обчисленні функцій для вікна {window}: {e}")
                        return {}

                # Паралельне обчислення для різних розмірів вікон
                window_features_dict = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_window = {executor.submit(compute_window_features, window): window for window in
                                        window_sizes}
                    for future in concurrent.futures.as_completed(future_to_window):
                        window_features = future.result()
                        window_features_dict.update(window_features)

                # Додавання обчислених функцій до основного DataFrame
                for col, series in window_features_dict.items():
                    features[col] = series

            else:
                # Послідовне обчислення для меншої кількості вікон
                for window in window_sizes:
                    # Додавання історичної волатильності
                    features[f'hist_vol_{window}d'] = self.calculate_historical_volatility(
                        ohlc_data['close'], window=window)

                    # Додавання волатильності Паркінсона
                    features[f'park_vol_{window}d'] = self.calculate_parkinson_volatility(
                        ohlc_data, window=window)

                    # Додавання відносної волатильності
                    moving_avg_vol = features[f'hist_vol_{window}d'].rolling(window=window * 2).mean()
                    features[f'rel_vol_{window}d'] = features[f'hist_vol_{window}d'] / moving_avg_vol

                    # Додавання волатильності волатильності
                    features[f'vol_of_vol_{window}d'] = features[f'hist_vol_{window}d'].rolling(window=window).std()

                    # Тренд волатильності
                    features[f'vol_trend_{window}d'] = features[f'hist_vol_{window}d'].diff(window)

                    # Відношення діапазону High-Low до волатильності
                    hl_range = (ohlc_data['high'] - ohlc_data['low']) / ohlc_data['close']
                    features[f'hl_range_to_vol_{window}d'] = hl_range / features[f'hist_vol_{window}d']

            # Додавання ідентифікації режимів, якщо вказано
            if include_regimes:
                # Використання основної волатильності для виявлення режимів
                main_vol_col = next((col for col in features.columns if 'hist_vol_14d' in col), None)
                if main_vol_col and not features[main_vol_col].isna().all():
                    # Виявлення режимів та додавання як функцій
                    main_vol = features[main_vol_col].copy()
                    regimes = self.detect_volatility_regimes(main_vol, n_regimes=3)

                    # Оптимізація: векторизоване one-hot кодування
                    for i in range(3):
                        features[f'vol_regime_{i}'] = (regimes == i).astype(int)

                    # Додавання функцій перемикання режимів
                    regime_changes = regimes.diff().abs()
                    features['regime_change'] = regime_changes.replace({0: 0, 1: 1, 2: 1})
                    features['regime_change_lag'] = features['regime_change'].shift(1).fillna(0)

            # Додавання додаткових функцій із перетвореннями
            if 'hist_vol_14d' in features.columns:
                # Логарифмовані значення волатильності для лінеаризації
                features['log_vol'] = np.log1p(features['hist_vol_14d'])

                # Нормалізована волатильність (z-score)
                roll_mean = features['hist_vol_14d'].rolling(window=30).mean()
                roll_std = features['hist_vol_14d'].rolling(window=30).std()
                features['vol_zscore'] = (features['hist_vol_14d'] - roll_mean) / roll_std

                # Індикатор аномальної волатильності (>2 стандартних відхилень)
                features['vol_outlier'] = (np.abs(features['vol_zscore']) > 2).astype(int)

            logger.info(f"Успішно підготовлено {len(features.columns)} функцій волатильності для ML")
            return features

        except Exception as e:
            logger.error(f"Помилка при підготовці функцій волатильності для ML: {e}")
            return pd.DataFrame(index=ohlc_data.index)

    def save_volatility_analysis_to_db(self, symbol, timeframe, volatility_data, model_data=None, regime_data=None,
                                       features_data=None, cross_asset_data=None):
        """Збереження аналізу волатильності до бази даних (оптимізовано)"""
        try:
            logger.info(f"Збереження аналізу волатильності для {symbol} на таймфреймі {timeframe}")
            results = {}
            success = True

            # Оптимізація: паралельне збереження різних типів даних
            if self.use_parallel:
                save_tasks = []

                # 1. Збереження основних метрик волатильності
                if volatility_data is not None and not volatility_data.empty:
                    save_tasks.append(('metrics', lambda: self.db_manager.save_volatility_metrics(
                        symbol=symbol,
                        timeframe=timeframe,
                        metrics=volatility_data
                    )))

                # 2. Збереження моделей волатильності (GARCH тощо)
                if model_data is not None:
                    save_tasks.append(('model', lambda: self.db_manager.save_volatility_model(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_data.get('name', 'garch'),
                        parameters=model_data.get('params', {}),
                        forecast_data=model_data.get('forecast'),
                        model_stats=model_data.get('stats', {})
                    )))

                # 3. Збереження даних режиму
                if regime_data is not None:
                    save_tasks.append(('regime', lambda: self.db_manager.save_volatility_regime(
                        symbol=symbol,
                        timeframe=timeframe,
                        regime_data=regime_data.get('regimes'),
                        regime_method=regime_data.get('method', 'kmeans'),
                        regime_params=regime_data.get('params', {})
                    )))

                # 4. Збереження ML функцій
                if features_data is not None and not features_data.empty:
                    save_tasks.append(('features', lambda: self.db_manager.save_volatility_features(
                        symbol=symbol,
                        timeframe=timeframe,
                        features_data=features_data
                    )))

                # 5. Збереження даних крос-активної волатильності
                if cross_asset_data is not None and not cross_asset_data.empty:
                    save_tasks.append(('cross_asset', lambda: self.db_manager.save_cross_asset_volatility(
                        base_symbol=symbol,
                        timeframe=timeframe,
                        correlation_data=cross_asset_data
                    )))

                # Паралельне виконання завдань збереження
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(len(save_tasks), self.max_workers)) as executor:
                    future_to_task = {executor.submit(task_func): task_name for task_name, task_func in save_tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        task_name = future_to_task[future]
                        try:
                            task_success = future.result()
                            results[task_name] = task_success
                            success = success and task_success
                            logger.info(f"Збережено {task_name}: {task_success}")
                        except Exception as e:
                            logger.error(f"Помилка при збереженні {task_name}: {e}")
                            results[task_name] = False
                            success = False
            else:
                # Послідовне збереження
                # 1. Збереження основних метрик волатильності
                if volatility_data is not None and not volatility_data.empty:
                    metrics_success = self.db_manager.save_volatility_metrics(
                        symbol=symbol,
                        timeframe=timeframe,
                        metrics_data=volatility_data
                    )
                    results['metrics'] = metrics_success
                    success = success and metrics_success
                    logger.info(f"Збережено метрики волатильності: {metrics_success}")

                # 2. Збереження моделей волатильності (GARCH тощо)
                if model_data is not None:
                    model_success = self.db_manager.save_volatility_model(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_name=model_data.get('name', 'garch'),
                        model_params=model_data.get('params', {}),
                        forecast_data=model_data.get('forecast'),
                        model_stats=model_data.get('stats', {})
                    )
                    results['model'] = model_success
                    success = success and model_success
                    logger.info(f"Збережено модель волатильності: {model_success}")

                # 3. Збереження даних режиму
                if regime_data is not None:
                    regime_success = self.db_manager.save_volatility_regime(
                        symbol=symbol,
                        timeframe=timeframe,
                        regime_data=regime_data.get('regimes'),
                        regime_method=regime_data.get('method', 'kmeans'),
                        regime_params=regime_data.get('params', {})
                    )
                    results['regime'] = regime_success
                    success = success and regime_success
                    logger.info(f"Збережено режими волатильності: {regime_success}")

                # 4. Збереження ML функцій
                if features_data is not None and not features_data.empty:
                    features_success = self.db_manager.save_volatility_features(
                        symbol=symbol,
                        timeframe=timeframe,
                        features_data=features_data
                    )
                    results['features'] = features_success
                    success = success and features_success
                    logger.info(f"Збережено функції волатильності: {features_success}")

                # 5. Збереження даних крос-активної волатильності
                if cross_asset_data is not None and not cross_asset_data.empty:
                    cross_asset_success = self.db_manager.save_cross_asset_volatility(
                        base_symbol=symbol,
                        timeframe=timeframe,
                        correlation=cross_asset_data
                    )
                    results['cross_asset'] = cross_asset_success
                    success = success and cross_asset_success
                    logger.info(f"Збережено крос-активну волатильність: {cross_asset_success}")

            return {'overall_success': success, 'detailed_results': results}

        except Exception as e:
            logger.error(f"Помилка при збереженні аналізу волатильності до бази даних: {e}")
            return {'overall_success': False, 'error': str(e)}

    def load_volatility_analysis_from_db(self, symbol, timeframe, start_date=None, end_date=None):
        """Завантаження аналізу волатильності з бази даних (оптимізовано)"""
        try:
            logger.info(f"Завантаження аналізу волатильності для {symbol} на таймфреймі {timeframe}")

            # Оптимізація: паралельне завантаження різних типів даних
            results = {}

            if self.use_parallel:
                load_tasks = [
                    ('metrics_data', lambda: self.db_manager.get_volatility_metrics(
                        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                    )),
                    ('model_data', lambda: self.db_manager.get_volatility_model(
                        symbol=symbol, timeframe=timeframe, model_name='garch'
                    )),
                    ('regime_data', lambda: self.db_manager.get_volatility_regime(
                        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                    )),
                    ('features_data', lambda: self.db_manager.get_volatility_features(
                        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                    )),
                    ('cross_asset_data', lambda: self.db_manager.get_cross_asset_volatility(
                        base_symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                    ))
                ]

                # Паралельне виконання завдань завантаження
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(load_tasks)) as executor:
                    future_to_task = {executor.submit(task_func): task_name for task_name, task_func in load_tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        task_name = future_to_task[future]
                        try:
                            results[task_name] = future.result()
                        except Exception as e:
                            logger.error(f"Помилка при завантаженні {task_name}: {e}")
                            results[task_name] = None
            else:
                # Послідовне завантаження
                # 1. Завантаження метрик волатильності
                results['metrics_data'] = self.db_manager.get_volatility_metrics(
                    symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                )

                # 2. Завантаження даних моделі волатильності (за замовчуванням GARCH)
                results['model_data'] = self.db_manager.get_volatility_model(
                    symbol=symbol, timeframe=timeframe, model_name='garch'
                )

                # 3. Завантаження даних режиму
                results['regime_data'] = self.db_manager.get_volatility_regime(
                    symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                )

                # 4. Завантаження ML функцій
                results['features_data'] = self.db_manager.get_volatility_features(
                    symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                )

                # 5. Завантаження крос-активної волатильності
                results['cross_asset_data'] = self.db_manager.get_cross_asset_volatility(
                    base_symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                )

            # Об'єднання всіх результатів у комплексний об'єкт аналізу
            analysis_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'volatility_metrics': results.get('metrics_data'),
                'model_data': results.get('model_data'),
                'regime_data': results.get('regime_data'),
                'features_data': results.get('features_data'),
                'cross_asset_data': results.get('cross_asset_data'),
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }

            # Додавання підсумкової інформації, якщо дані метрик доступні
            metrics_data = analysis_results['volatility_metrics']
            if metrics_data is not None and not metrics_data.empty:
                # Оптимізація: використання векторизованого пошуку стовпців
                vol_cols = [col for col in metrics_data.columns if 'hist_vol_14d' in col]
                main_vol_col = vol_cols[0] if vol_cols else None

                if main_vol_col:
                    # Оптимізація: векторизовані обчислення статистики
                    summary = {
                        'avg_volatility': metrics_data[main_vol_col].mean(),
                        'current_volatility': metrics_data[main_vol_col].iloc[-1] if not metrics_data.empty else None,
                        'volatility_trend': 'increasing' if (len(metrics_data) >= 7 and
                                                             metrics_data[main_vol_col].iloc[-1] >
                                                             metrics_data[main_vol_col].iloc[-7])
                        else 'decreasing' if len(metrics_data) >= 7 else 'unknown',
                        'max_volatility': metrics_data[main_vol_col].max(),
                        'min_volatility': metrics_data[main_vol_col].min(),
                        'volatility_of_volatility': metrics_data[main_vol_col].std() / metrics_data[main_vol_col].mean()
                        if not metrics_data.empty and metrics_data[main_vol_col].mean() > 0 else None
                    }

                    # Додаткові корисні статистики
                    summary.update({
                        'percentile_90': metrics_data[main_vol_col].quantile(0.9),
                        'percentile_10': metrics_data[main_vol_col].quantile(0.1),
                        'days_above_average': (metrics_data[main_vol_col] > summary['avg_volatility']).sum(),
                        'current_percentile': percentileofscore(metrics_data[main_vol_col].dropna(),
                                                                summary['current_volatility'])
                        if summary['current_volatility'] is not None else None
                    })

                    analysis_results['summary'] = summary

            logger.info(f"Успішно завантажено аналіз волатильності для {symbol}")
            return analysis_results

        except Exception as e:
            logger.error(f"Помилка при завантаженні аналізу волатильності з бази даних: {e}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e)
            }

    def analyze_crypto_market_conditions(self, symbols=['BTC', 'ETH', 'SOL'], timeframe='1d', window=14):

        try:
            volatilities = {}

            # Get volatility for each symbol
            for symbol in symbols:
                data = self.db_manager.get_klines(f"{symbol}USDT", timeframe=timeframe)
                vol = self.calculate_historical_volatility(data['close'], window=window)
                volatilities[symbol] = vol

            # Convert to DataFrame
            vol_df = pd.DataFrame(volatilities)

            # Calculate market-wide metrics
            market_vol = vol_df.mean(axis=1)  # Average volatility across assets
            vol_dispersion = vol_df.std(axis=1)  # Dispersion in volatility
            vol_correlation = vol_df.corr().mean().mean()  # Average correlation

            # Get market phases using helper function
            market_phases = self.get_market_phases(vol_df)
            current_phase = market_phases.iloc[-1] if not market_phases.empty else None

            # Determine if in volatility regime shift
            regime_shifts = {}
            for symbol in symbols:
                regimes = self.detect_volatility_regimes(volatilities[symbol])
                # Check if regime changed in last 3 periods
                recent_changes = regimes.diff().iloc[-3:].abs().sum()
                regime_shifts[symbol] = recent_changes > 0

            # Return consolidated market analysis
            return {
                'average_market_vol': market_vol.iloc[-1],
                'vol_trend_30d': (market_vol.iloc[-1] / market_vol.iloc[-30]) - 1 if len(market_vol) >= 30 else None,
                'vol_dispersion': vol_dispersion.iloc[-1],
                'vol_correlation': vol_correlation,
                'market_phase': current_phase,
                'regime_shifts': regime_shifts,
                'high_vol_assets': [s for s, v in vol_df.iloc[-1].items() if v > vol_df.iloc[-1].mean()],
                'low_vol_assets': [s for s, v in vol_df.iloc[-1].items() if v < vol_df.iloc[-1].mean()]
            }

        except Exception as e:
            logger.error(f"Error analyzing crypto market conditions: {e}")
            return None

    def run_full_volatility_analysis(self, symbol, timeframe='1d', save_to_db=True):

        try:
            logger.info(f"Running full volatility analysis for {symbol} on {timeframe} timeframe")

            # Load data
            data = self.db_manager.get_klines(symbol, timeframe=timeframe)

            # Calculate returns
            data['returns'] = data['close'].pct_change()

            # Calculate various volatility metrics
            volatility = {}

            # Historical volatility for different windows
            for window in [7, 14, 30, 60]:
                volatility[f'hist_vol_{window}d'] = self.calculate_historical_volatility(
                    data['close'], window=window)

            # Parkinson and other volatility measures
            volatility['parkinson_vol'] = self.calculate_parkinson_volatility(data)
            volatility['gk_vol'] = self.calculate_garman_klass_volatility(data)
            volatility['yz_vol'] = self.calculate_yang_zhang_volatility(data)

            # Convert to DataFrame
            vol_df = pd.DataFrame(volatility)

            # Detect volatility regimes
            vol_df['regime'] = self.detect_volatility_regimes(vol_df['hist_vol_14d'])

            # Create regime data dictionary for database
            regime_data = {
                'regimes': vol_df['regime'],
                'method': 'kmeans',
                'params': {
                    'n_regimes': 3,
                    'feature': 'hist_vol_14d'
                }
            }

            # Analyze volatility clustering
            acf_data = self.analyze_volatility_clustering(data['returns'])

            # Calculate risk metrics
            risk_metrics = self.calculate_volatility_risk_metrics(
                data['returns'], vol_df['hist_vol_14d'])

            # Identify volatility breakouts
            vol_df['breakout'] = self.identify_volatility_breakouts(vol_df['hist_vol_14d'])

            # Get seasonality patterns
            seasonality = {}
            seasonality['dow'] = self.extract_seasonality_in_volatility(vol_df['hist_vol_14d'], period=7)
            seasonality['month'] = self.extract_seasonality_in_volatility(vol_df['hist_vol_14d'], period=12)

            # Fit GARCH model
            garch_model, garch_forecast = self.fit_garch_model(data['returns'])

            # Extract forecast values if model was successfully fit
            forecast_values = None
            if garch_model is not None:
                forecast_values = garch_forecast.variance.iloc[-1].values

            # Create model data dictionary for database
            model_data = {
                'name': 'garch',
                'params': {
                    'p': 1,
                    'q': 1,
                    'mean': 'Zero',
                    'vol': 'GARCH'
                },
                'forecast': forecast_values,
                'stats': {
                    'aic': garch_model.aic if garch_model is not None else None,
                    'bic': garch_model.bic if garch_model is not None else None
                }
            }

            # Calculate volatility impulse response
            impulse_response = self.volatility_impulse_response(data['returns'])

            # Prepare features for ML models
            ml_features = self.prepare_volatility_features_for_ml(data)

            # Get market-wide conditions for context
            market_conditions = self.analyze_crypto_market_conditions(
                symbols=[symbol, 'BTC', 'ETH'], timeframe=timeframe)

            # Get cross-asset correlation data
            cross_asset_symbols = ['BTC', 'ETH', 'BNB', 'XRP']
            asset_dict = {}
            for asset in cross_asset_symbols:
                asset_data = self.db_manager.get_klines(f"{asset}USDT", timeframe=timeframe)
                if asset_data is not None and not asset_data.empty:
                    asset_dict[asset] = asset_data['close']

            cross_asset_vol = self.analyze_cross_asset_volatility(asset_dict) if asset_dict else None

            # Combine all results
            analysis_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'volatility_data': vol_df,
                'latest_volatility': {
                    'hist_vol_14d': vol_df['hist_vol_14d'].iloc[-1] if not vol_df.empty else None,
                    'parkinson': vol_df['parkinson_vol'].iloc[-1] if not vol_df.empty else None,
                    'garman_klass': vol_df['gk_vol'].iloc[-1] if not vol_df.empty else None,
                    'yang_zhang': vol_df['yz_vol'].iloc[-1] if not vol_df.empty else None
                },
                'current_regime': vol_df['regime'].iloc[-1] if not vol_df.empty else None,
                'volatility_clustering': {
                    'significant_lags': acf_data[acf_data['autocorrelation'] > 0.1]['lag'].tolist(),
                    'max_autocorrelation': acf_data['autocorrelation'].max()
                },
                'risk_metrics': risk_metrics,
                'seasonality': seasonality,
                'recent_breakouts': vol_df['breakout'].iloc[-30:].sum() if len(vol_df) >= 30 else 0,
                'garch_forecast': forecast_values,
                'impulse_response': impulse_response,
                'market_conditions': market_conditions
            }

            # Calculate summary stats
            analysis_results['summary'] = {
                'avg_volatility': vol_df['hist_vol_14d'].mean(),
                'volatility_trend': 'increasing' if vol_df['hist_vol_14d'].iloc[-1] > vol_df['hist_vol_14d'].iloc[
                    -7] else 'decreasing',
                'regime_changes': vol_df['regime'].diff().abs().sum(),
                'volatility_of_volatility': vol_df['hist_vol_14d'].rolling(window=14).std().iloc[-1] if len(
                    vol_df) >= 14 else None,
                'current_vs_historical': vol_df['hist_vol_14d'].iloc[-1] / vol_df[
                    'hist_vol_14d'].mean() if not vol_df.empty else None
            }

            # Save to database if requested
            if save_to_db:
                logger.info(f"Saving volatility analysis for {symbol} to database")
                save_success = self.save_volatility_analysis_to_db(
                    symbol=symbol,
                    timeframe=timeframe,
                    volatility_data=vol_df,
                    model_data=model_data,
                    regime_data=regime_data,
                    features_data=ml_features,
                    cross_asset_data=cross_asset_vol
                )
                analysis_results['saved_to_db'] = save_success

            # Generate reports and visualizations
            self._generate_volatility_report(symbol, timeframe, analysis_results)

            return analysis_results

        except Exception as e:
            logger.error(f"Error in full volatility analysis for {symbol}: {e}")
            # Return partial results if available
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e),
                'partial_results': locals().get('vol_df', None)
            }

    def _generate_volatility_report(self, symbol, timeframe, analysis_results):

        try:
            vol_df = analysis_results['volatility_data']

            # Create plots directory if it doesn't exist
            import os
            os.makedirs('reports/volatility', exist_ok=True)

            # Plot volatility metrics
            plt.figure(figsize=(12, 8))
            for col in ['hist_vol_14d', 'parkinson_vol', 'gk_vol', 'yz_vol']:
                if col in vol_df.columns:
                    plt.plot(vol_df.index, vol_df[col], label=col)
            plt.title(f"{symbol} Volatility Metrics - {timeframe}")
            plt.legend()
            plt.savefig(f"reports/volatility/{symbol}_{timeframe}_volatility_metrics.png")

            # Plot volatility regimes
            plt.figure(figsize=(12, 6))
            plt.plot(vol_df.index, vol_df['hist_vol_14d'], label='Historical Vol (14d)')
            plt.scatter(vol_df.index, vol_df['hist_vol_14d'], c=vol_df['regime'], cmap='viridis', label='Regimes')
            plt.title(f"{symbol} Volatility Regimes - {timeframe}")
            plt.colorbar(label='Regime')
            plt.legend()
            plt.savefig(f"reports/volatility/{symbol}_{timeframe}_volatility_regimes.png")

            # Plot seasonality
            if 'seasonality' in analysis_results and 'dow' in analysis_results['seasonality']:
                plt.figure(figsize=(10, 6))
                analysis_results['seasonality']['dow'].plot(kind='bar')
                plt.title(f"{symbol} Day-of-Week Volatility Seasonality")
                plt.savefig(f"reports/volatility/{symbol}_{timeframe}_dow_seasonality.png")

            logger.info(f"Generated volatility report for {symbol}")

        except Exception as e:
            logger.error(f"Error generating volatility report: {e}")