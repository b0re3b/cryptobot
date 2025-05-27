import concurrent.futures

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import percentileofscore
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import acf

from data.db import DatabaseManager
from featureengineering.feature_engineering import FeatureEngineering
from utils.logger import CryptoLogger


class VolatilityAnalysis:

    def __init__(self, use_parallel=True, max_workers=4):
        self.db_manager = DatabaseManager()
        self.volatility_models = {}
        self.regime_models = {}
        self.feature_engineer = FeatureEngineering()
        self.logger = CryptoLogger('volatility_analysis')
        self.use_parallel = use_parallel
        self.max_workers = max_workers

    def _calc_log_returns(self, prices_tuple):
        """
            Обчислює логарифмічні прибутки для послідовності цін.

            Використовується для аналізу фінансових часових рядів, де лог-прибутки дають змогу
            стабілізувати дисперсію та використовуються у фінансових моделях.

            Args:
                prices_tuple (tuple або list): Послідовність цін (наприклад, закриття).

            Returns:
                np.ndarray: Масив логарифмічних прибутків між послідовними цінами.
            """
        prices = np.array(prices_tuple)
        return np.log(prices[1:] / prices[:-1])

    def _parallel_process(self, func, items, *args, **kwargs):
        """
            Виконує обчислення над колекцією об'єктів у паралельному режимі.

            Якщо ввімкнено `self.use_parallel` та кількість елементів перевищує поріг,
            обробка відбувається багатопотоково з використанням ThreadPoolExecutor.

            Args:
                func (Callable): Функція, яка буде застосована до кожного елемента.
                items (Iterable): Список елементів для обробки.
                *args: Додаткові позиційні аргументи для функції.
                **kwargs: Додаткові іменовані аргументи для функції.

            Returns:
                dict: Словник результатів обробки, де ключ — елемент, значення — результат або None при помилці.
            """
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
                    self.logger.error(f"Помилка при паралельній обробці для {item}: {e}")
                    results[item] = None
        return results

    def _safe_convert_to_float(self, series):
        """
            Безпечно перетворює вхідні дані до типу float64.

            Призначено для уніфікації типів перед числовими обчисленнями.

            Args:
                series (pd.Series, np.ndarray або список): Вхідні дані.

            Returns:
                np.ndarray або pd.Series: Об'єкт того ж типу, але з типом float64.
            """
        if isinstance(series, pd.Series):
            return series.astype(float)
        elif isinstance(series, np.ndarray):
            return series.astype(float)
        else:
            return np.array(series, dtype=float)

    def get_market_phases(self, volatility_data, lookback_window=90, n_regimes=4):
        """
           Визначає фази ринку на основі кластеризації ознак волатильності.

           Аналізує історичну волатильність по всіх активах та класифікує її
           на кілька режимів (фаз) за допомогою KMeans.

           Args:
               volatility_data (pd.DataFrame): Таблиця волатильності з індексом часу та колонками по активах.
               lookback_window (int): Кількість останніх днів для аналізу.
               n_regimes (int): Кількість режимів (кластерів), на які розділяється ринок.

           Returns:
               pd.Series: Серія з фазами ринку (Low Vol, High Vol тощо), вирівнана по індексу вхідних даних.
           """
        try:
            self.logger.info(f"Визначення фаз ринку за допомогою {n_regimes} режимів")

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
                self.logger.warning("Недостатньо даних для визначення фаз ринку")
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
            full_phase_series = full_phase_series.astype('object')
            full_phase_series = full_phase_series.ffill().bfill()  # Заповнення в обидва боки

            # Зберігаємо модель для подальшого використання
            self.regime_models[f"market_phases_{n_regimes}"] = {
                'model': kmeans,
                'scaler': scaler,
                'features': list(features.columns),
                'mapping': phase_mapping
            }

            self.logger.info(f"Успішно визначено {n_regimes} фази ринку")
            return full_phase_series

        except Exception as e:
            self.logger.error(f"Помилка при визначенні фаз ринку: {e}")
            return pd.Series(index=volatility_data.index)

    def calculate_historical_volatility(self, price_data, window=14, trading_periods=365, annualize=True):
        """
          Розраховує історичну (реалізовану) волатильність на основі логарифмічних прибутків.

          Історична волатильність показує середнє відхилення відносних змін цін (лог-повернень) за певний період.

          Args:
              price_data (pd.Series або list або np.ndarray): Історичні ціни активу.
              window (int, optional): Розмір ковзного вікна для розрахунку стандартного відхилення. За замовчуванням 14.
              trading_periods (int, optional): Кількість торгових періодів на рік (для анулізації). За замовчуванням 365.
              annualize (bool, optional): Якщо True, масштабувати волатильність до річного значення.

          Returns:
              pd.Series: Ряд, що містить значення історичної волатильності.
          """
        try:
            # Валідація параметра вікна
            if not isinstance(window, int) or window <= 0:
                raise ValueError("Параметр 'window' має бути додатним цілим числом")

            # Конвертація у Series, якщо потрібно
            price_data = pd.Series(price_data) if not isinstance(price_data, pd.Series) else price_data

            # Конвертація значень у float (у разі decimal або object dtype)
            price_data = self._safe_convert_to_float(price_data)

            # Обробка нульових або від'ємних значень
            if (price_data <= 0).any():
                self.logger.warning("Виявлено нульові або від’ємні ціни — буде виконано заповнення")
                price_data = price_data.replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
                price_data = price_data.ffill().bfill()

            # Повторна перевірка: якщо залишилися некоректні значення
            if (price_data <= 0).any():
                self.logger.error("Не вдалося усунути всі нульові або від’ємні значення в price_data")
                return pd.Series(index=price_data.index, dtype=float)

            # Розрахунок логарифмічних прибутків
            log_returns = np.log(price_data / price_data.shift(1)).dropna()

            # Розрахунок ковзної стандартної девіації (волатильність)
            rolling_vol = log_returns.rolling(window=window).std()

            # Анулізація (переведення до річного значення)
            if annualize:
                rolling_vol *= np.sqrt(trading_periods)

            self.logger.debug(f"Історична волатильність розрахована для {len(rolling_vol)} точок")
            return rolling_vol

        except Exception as e:
            self.logger.error(f"Помилка при обчисленні історичної волатильності: {e}")
            return pd.Series(index=price_data.index if isinstance(price_data, pd.Series) else range(len(price_data)),
                             dtype=float)

    def calculate_parkinson_volatility(self, ohlc_data, window=14, trading_periods=365):
        """
           Розраховує волатильність Паркінсона на основі діапазону "high-low".

           Метод Паркінсона враховує лише максимальні та мінімальні ціни для більш точної оцінки волатильності
           у разі відсутності значного "gap" між відкриттям і закриттям.

           Args:
               ohlc_data (pd.DataFrame): Дані OHLC (повинні містити колонки 'high' і 'low').
               window (int, optional): Ковзне вікно для обчислення середнього. За замовчуванням 14.
               trading_periods (int, optional): Кількість торгових днів у році (для анулізації).

           Returns:
               pd.Series: Ряд з волатильністю Паркінсона за кожну дату.
           """
        # Перевірка, чи є ohlc_data DataFrame
        if not isinstance(ohlc_data, pd.DataFrame):
            self.logger.error("Дані OHLC повинні бути DataFrame")
            return pd.Series()

        try:
            # Конвертація в float для безпечних обчислень
            high = self._safe_convert_to_float(ohlc_data['high'])
            low = self._safe_convert_to_float(ohlc_data['low'])

            # Перевірка на нульові або від'ємні значення
            mask = (high > 0) & (low > 0)
            if (~mask).any():
                self.logger.warning("Знайдено нульові або від'ємні значення в 'high' або 'low' - ці точки будуть пропущені")

            # Векторизований розрахунок нормалізованого діапазону high-low з обробкою проблемних значень
            hl_range = np.zeros(len(high))
            hl_range[mask] = np.log(high[mask] / low[mask])

            parkinson = pd.Series(hl_range ** 2 / (4 * np.log(2)), index=ohlc_data.index)

            # Обчислення волатильності та анулізація
            rolling_parkinson = np.sqrt(parkinson.rolling(window=window).mean() * trading_periods)

            return rolling_parkinson
        except KeyError:
            self.logger.error("Дані OHLC повинні містити стовпці 'high' і 'low'")
            return pd.Series()

    def calculate_garman_klass_volatility(self, ohlc_data, window=14, trading_periods=365):
        """
            Розраховує волатильність за формулою Гарман-Класс.

            Метод Гарман-Класс поєднує інформацію про діапазон (high/low) та співвідношення між цінами відкриття і закриття,
            що робить його точнішим, ніж історична або Паркінсона волатильність.

            Args:
                ohlc_data (pd.DataFrame): Дані OHLC, які містять колонки 'high', 'low', 'open' та 'close'.
                window (int, optional): Розмір ковзного вікна для обчислення волатильності. За замовчуванням 14.
                trading_periods (int, optional): Кількість торгових днів у році (для анулізації результату).

            Returns:
                pd.Series: Ряд з волатильністю Гарман-Класс по кожному періоду.
            """
        try:
            # Конвертація в float для безпечних обчислень
            high = self._safe_convert_to_float(ohlc_data['high'])
            low = self._safe_convert_to_float(ohlc_data['low'])
            close = self._safe_convert_to_float(ohlc_data['close'])
            open_price = self._safe_convert_to_float(ohlc_data['open'])

            # Перевірка на нульові або від'ємні значення
            mask = (high > 0) & (low > 0) & (close > 0) & (open_price > 0)
            if (~mask).any():
                self.logger.warning("Знайдено нульові або від'ємні значення в OHLC даних - ці точки будуть пропущені")

            # Створення масивів для результатів
            log_hl = np.zeros(len(high))
            log_co = np.zeros(len(high))

            # Векторизоване обчислення компонентів тільки для валідних даних
            log_hl[mask] = np.log(high[mask] / low[mask]) ** 2 * 0.5
            valid_co = (close > 0) & (open_price > 0)
            log_co[valid_co] = np.log(close[valid_co] / open_price[valid_co]) ** 2 * (2 * np.log(2) - 1)

            # Комбінування компонентів
            gk = pd.Series(log_hl - log_co, index=ohlc_data.index)

            # Обчислення волатильності та анулізація
            rolling_gk = np.sqrt(gk.rolling(window=window).mean() * trading_periods)

            return rolling_gk
        except KeyError:
            self.logger.error("Дані OHLC повинні містити стовпці 'high', 'low', 'close' і 'open'")
            return pd.Series()

    def calculate_yang_zhang_volatility(self, ohlc_data, window=14, trading_periods=365):
        """
            Розраховує волатильність Янг-Чжан (Yang-Zhang Volatility), яка об'єднує overnight,
            intraday та Rogers-Satchell компоненти для точнішої оцінки волатильності.

            Метод Yang-Zhang вважається більш надійним за інші (наприклад, Garman-Klass або історичну волатильність),
            оскільки враховує:
                - нічну волатильність (від закриття до відкриття),
                - денну волатильність (відкриття до закриття),
                - інформацію про внутрішньоденні коливання (high/low).

            Args:
                ohlc_data (pd.DataFrame): OHLC дані, які повинні містити стовпці 'open', 'high', 'low', 'close'.
                window (int, optional): Розмір ковзного вікна для обчислення. За замовчуванням 14.
                trading_periods (int, optional): Кількість торгових періодів у році для анулізації. За замовчуванням 365.

            Returns:
                pd.Series: Ряд з розрахованими значеннями волатильності за методом Yang-Zhang.
            """
        try:
            # Конвертація в float для безпечних обчислень
            high = self._safe_convert_to_float(ohlc_data['high'])
            low = self._safe_convert_to_float(ohlc_data['low'])
            close = self._safe_convert_to_float(ohlc_data['close'])
            open_price = self._safe_convert_to_float(ohlc_data['open'])
            close_prev = close.shift(1)

            # Перевірка на нульові або від'ємні значення
            valid_data = (high > 0) & (low > 0) & (close > 0) & (open_price > 0) & (close_prev > 0)
            if (~valid_data).any():
                self.logger.warning("Знайдено нульові або від'ємні значення в OHLC даних - ці точки будуть пропущені")

            # Маскування невалідних даних як NaN для безпечних обчислень
            high_masked = pd.Series(np.where(valid_data, high, np.nan), index=ohlc_data.index)
            low_masked = pd.Series(np.where(valid_data, low, np.nan), index=ohlc_data.index)
            close_masked = pd.Series(np.where(valid_data, close, np.nan), index=ohlc_data.index)
            open_masked = pd.Series(np.where(valid_data, open_price, np.nan), index=ohlc_data.index)
            close_prev_masked = pd.Series(np.where(valid_data, close_prev, np.nan), index=ohlc_data.index)

            # Нічна волатильність (close to open)
            overnight_returns = np.log(open_masked / close_prev_masked)
            overnight_vol = overnight_returns.rolling(window=window).var()

            # Волатильність open-close
            open_close_returns = np.log(close_masked / open_masked)
            open_close_vol = open_close_returns.rolling(window=window).var()

            # Волатильність Rogers-Satchell
            log_ho = np.log(high_masked / open_masked)
            log_lo = np.log(low_masked / open_masked)
            log_hc = np.log(high_masked / close_masked)
            log_lc = np.log(low_masked / close_masked)

            rs_vol = log_ho * (log_ho - log_lo) + log_lc * (log_lc - log_hc)
            rs_vol = rs_vol.rolling(window=window).mean()

            # Обчислення волатильності Yang-Zhang з k=0.34 (рекомендоване значення)
            k = 0.34
            yang_zhang = overnight_vol + k * open_close_vol + (1 - k) * rs_vol

            # Анулізація
            yang_zhang = np.sqrt(yang_zhang * trading_periods)

            return yang_zhang
        except KeyError:
            self.logger.error("Дані OHLC повинні містити стовпці 'high', 'low', 'close' і 'open'")
            return pd.Series()

    def fit_garch_model(self, returns, p=1, q=1, model_type='GARCH', dist='normal', mean='Zero',
                        horizon=30, options=None):
        """
        Підгонка моделі GARCH та отримання прогнозу волатильності.

        Параметри:
            returns (pd.Series/array-like): Серія прибутків
            p (int): Кількість лагів для GARCH компоненти
            q (int): Кількість лагів для ARCH компоненти
            model_type (str): Тип моделі ('GARCH', 'EGARCH', 'GJR-GARCH')
            dist (str): Розподіл інновацій ('normal', 't', 'skewt')
            mean (str): Тип середнього ('Zero', 'Constant', 'AR', 'LS')
            horizon (int): Горизонт прогнозування
            options (dict): Додаткові параметри для оптимізації моделі

        Повертає:
            tuple: (fitted_model, forecast) або (None, None) у разі помилки
        """
        try:
            # Конвертація вхідних даних у безпечний формат
            returns = self._safe_convert_to_float(pd.Series(returns))

            if len(returns) < max(p, q) * 10:  # Мінімальна кількість спостережень
                self.logger.warning(f"Недостатньо даних для підгонки {model_type}({p},{q}) моделі")
                return None, None

            self.logger.info(f"Підгонка {model_type}({p},{q}) моделі з розподілом {dist} та середнім {mean}")

            # Налаштування моделі згідно з параметрами
            model_kwargs = {
                'vol': model_type,
                'p': p,
                'q': q,
                'dist': dist,
                'mean': mean
            }

            # Специфічні параметри для різних типів моделей
            if model_type == 'EGARCH':
                model_kwargs['o'] = 1  # Асиметричний ефект для EGARCH
            elif model_type == 'GJR-GARCH':
                model_kwargs['o'] = 1  # Асиметричний ефект для GJR-GARCH

            model = arch_model(returns, **model_kwargs)

            # Параметри оптимізації
            fit_options = {
                'disp': 'off',
                'maxiter': 500,
                'tol': 1e-8,
                'starting_values': None
            }
            if options:
                fit_options.update(options)

            # Підгонка моделі
            fitted_model = model.fit(**fit_options)

            # Прогнозування волатильності
            forecast = fitted_model.forecast(horizon=horizon)

            # Збереження моделі для подальшого використання
            model_key = f"{model_type}_{p}_{q}_{dist}_{mean}"
            self.volatility_models[model_key] = {
                'model': fitted_model,
                'last_fit_date': pd.Timestamp.now(),
                'params': model_kwargs
            }

            self.logger.info(
                f"Модель {model_type}({p},{q}) успішно підігнана. "
                f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}"
            )

            return fitted_model, forecast

        except Exception as e:
            self.logger.error(
                f"Помилка підгонки {model_type}({p},{q}) моделі: {str(e)}",
                exc_info=True
            )
            return None, None

    def detect_volatility_regimes(self, volatility_series, n_regimes=3, method='kmeans'):
        """
           Виявляє режими волатильності (наприклад, низька, середня, висока) за допомогою кластеризації або порогів.

           Доступні методи:
               - 'kmeans': кластеризація KMeans на основі значень волатильності.
               - 'threshold': використання процентильних порогів.
               - 'quantile': розбиття за квантилями (за допомогою pd.qcut).

           Args:
               volatility_series (pd.Series): Серія з обчисленою волатильністю.
               n_regimes (int): Кількість кластерів/режимів. За замовчуванням 3.
               method (str): Метод класифікації режимів ('kmeans', 'threshold', 'quantile').

           Returns:
               pd.Series: Серія з мітками режимів для кожної точки часу (0 = низький, ..., n-1 = високий).
           """
        try:
            self.logger.info(f"Виявлення {n_regimes} режимів волатильності за методом {method}")

            # Конвертація в float для безпечних обчислень
            volatility_series = self._safe_convert_to_float(volatility_series)

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

            self.logger.info(f"Успішно виявлено {n_regimes} режимів волатильності")
            return regime_series

        except Exception as e:
            self.logger.error(f"Помилка при виявленні режимів волатильності: {e}")
            return None

    def analyze_volatility_clustering(self, returns, max_lag=30):
        """
           Аналізує наявність автокореляції в квадраті прибутків для виявлення кластеризації волатильності.

           Принцип:
               Якщо автокореляція квадратів прибутків позитивна — це ознака того, що
               волатильність схильна до кластера: після високої волатильності часто слідує висока, і навпаки.

           Args:
               returns (pd.Series): Ряд з прибутками (log або простими).
               max_lag (int): Максимальна кількість лагів для обчислення автокореляції.

           Returns:
               pd.DataFrame: Таблиця з колонками ['lag', 'autocorrelation'], що відображає значення ACF.
           """
        try:
            # Конвертація в float для безпечних обчислень
            returns = self._safe_convert_to_float(returns)

            # Обчислення квадратів прибутку
            squared_returns = returns ** 2

            # Обчислення автокореляції
            # Оптимізація: обробка пропусків перед обчисленням ACF
            clean_squared_returns = squared_returns.dropna()

            if len(clean_squared_returns) < max_lag + 1:
                self.logger.warning(f"Недостатньо даних для обчислення автокореляції з {max_lag} лагами")
                return pd.DataFrame({'lag': range(len(clean_squared_returns)), 'autocorrelation': 0})

            acf_values = acf(clean_squared_returns, nlags=max_lag, fft=True)  # FFT для швидкої обробки

            # Створення результуючого DataFrame
            acf_df = pd.DataFrame({
                'lag': range(len(acf_values)),
                'autocorrelation': acf_values
            })

            return acf_df
        except Exception as e:
            self.logger.error(f"Помилка при аналізі кластеризації волатильності: {e}")
            return pd.DataFrame({'lag': [0], 'autocorrelation': [0]})

    def calculate_volatility_risk_metrics(self, returns, volatility):
        """
           Розраховує ключові метрики ризику, пов’язані з волатильністю активу.

           Метрики:
               - VaR (Value at Risk) на 95% і 99% рівнях
               - CVaR (Conditional VaR) або очікувані збитки в зоні tail
               - Volatility of Volatility: середнє стандартне відхилення волатильності
               - Sharpe Ratio (анулізований)
               - Максимальне просідання (max drawdown)

           Args:
               returns (pd.Series): Серія з log-прибутками активу.
               volatility (pd.Series): Обчислена волатильність на тих самих індексах.

           Returns:
               Dict[str, float]: Словник з розрахованими метриками ризику.
           """
        try:
            self.logger.info("Обчислення метрик ризику волатильності")

            # Конвертація в float для безпечних обчислень
            returns = self._safe_convert_to_float(returns)
            volatility = self._safe_convert_to_float(volatility)

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
            self.logger.error(f"Помилка при обчисленні метрик ризику волатильності: {e}")
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
        """
            Порівнює декілька типів волатильності (історична, Паркінсона, Гарман-Класс, Янг-Чжанг) для заданих вікон.

            Для прискорення обчислень за великої кількості вікон або обсягів даних використовує паралелізм, якщо `self.use_parallel` увімкнено.

            Args:
                ohlc_data (pd.DataFrame): OHLC-дані (мають містити стовпці: 'open', 'high', 'low', 'close').
                windows (List[int]): Список довжин вікон для обчислення волатильності (у днях).

            Returns:
                pd.DataFrame: Таблиця з колонками різних метрик волатильності для кожного вікна.
            """
        try:
            self.logger.info(f"Порівняння метрик волатильності для вікон {windows}")

            result = pd.DataFrame(index=ohlc_data.index)

            # Конвертація OHLC даних у float для безпечних обчислень
            ohlc_data_float = ohlc_data.copy()
            for col in ['open', 'high', 'low', 'close']:
                if col in ohlc_data.columns:
                    ohlc_data_float[col] = self._safe_convert_to_float(ohlc_data[col])

            # Обчислення прибутків
            ohlc_data_float['returns'] = ohlc_data_float['close'].pct_change()

            # Оптимізація: паралельне обчислення для різних вікон у великих наборах даних
            if self.use_parallel and len(windows) > 2:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Обчислення історичної волатильності
                    hist_vol_futures = {
                        executor.submit(self.calculate_historical_volatility,
                                        ohlc_data_float['close'], window=window): f'historical_{window}d'
                        for window in windows
                    }

                    # Обчислення волатильності Паркінсона
                    park_vol_futures = {
                        executor.submit(self.calculate_parkinson_volatility,
                                        ohlc_data_float, window=window): f'parkinson_{window}d'
                        for window in windows
                    }

                    # Обчислення волатильності Гарман-Класс
                    gk_vol_futures = {
                        executor.submit(self.calculate_garman_klass_volatility,
                                        ohlc_data_float, window=window): f'gk_{window}d'
                        for window in windows
                    }

                    # Обчислення волатильності Янг Чжанг
                    yz_vol_futures = {
                        executor.submit(self.calculate_yang_zhang_volatility,
                                        ohlc_data_float, window=window): f'yz_{window}d'
                        for window in windows
                    }

                    # Збір результатів
                    for future_dict in [hist_vol_futures, park_vol_futures, gk_vol_futures, yz_vol_futures]:
                        for future, col_name in future_dict.items():
                            try:
                                result[col_name] = future.result()
                            except Exception as e:
                                self.logger.error(f"Помилка при обчисленні {col_name}: {e}")
            else:
                # Послідовне обчислення для менших наборів даних
                for window in windows:
                    # Обчислення різних метрик волатильності
                    result[f'historical_{window}d'] = self.calculate_historical_volatility(
                        ohlc_data_float['close'], window=window)

                    result[f'parkinson_{window}d'] = self.calculate_parkinson_volatility(
                        ohlc_data_float, window=window)

                    result[f'gk_{window}d'] = self.calculate_garman_klass_volatility(
                        ohlc_data_float, window=window)

                    result[f'yz_{window}d'] = self.calculate_yang_zhang_volatility(
                        ohlc_data_float, window=window)

            self.logger.info("Успішно порівняно метрики волатильності")
            return result

        except Exception as e:
            self.logger.error(f"Помилка при порівнянні метрик волатильності: {e}")
            return pd.DataFrame(index=ohlc_data.index)

    def identify_volatility_breakouts(self, volatility_series, window=20, std_dev=2):
        """
           Виявляє прориви волатильності, коли поточне значення значно перевищує історичне середнє.

           Прорив вважається тоді, коли волатильність > середнє + N стандартних відхилень.

           Args:
               volatility_series (pd.Series): Часовий ряд волатильності.
               window (int): Розмір вікна для ковзного середнього і стандартного відхилення.
               std_dev (float): Множник стандартного відхилення для визначення порогу прориву.

           Returns:
               pd.Series: Булевий рядок із мітками True у точках прориву.
           """
        try:
            # Обчислення ковзного середнього та стандартного відхилення
            rolling_mean = volatility_series.rolling(window=window).mean()
            rolling_std = volatility_series.rolling(window=window).std()

            # Обчислення верхнього порогу
            upper_threshold = rolling_mean + std_dev * rolling_std

            # Виявлення проривів
            breakouts = volatility_series > upper_threshold

            self.logger.info(f"Виявлено {breakouts.sum()} проривів волатильності з {len(breakouts)} точок")
            return breakouts

        except Exception as e:
            self.logger.error(f"Помилка при виявленні проривів волатильності: {e}")
            return pd.Series(False, index=volatility_series.index)

    def analyze_cross_asset_volatility(self, asset_dict, window=14):
        """
            Аналізує кросс-активну волатильність: обчислює історичну волатильність кожного активу і між ними — кореляцію.

            Якщо активів багато, використовується паралельна обробка для прискорення.

            Args:
                asset_dict (Dict[str, pd.Series]): Словник, де ключ — назва активу, значення — часовий ряд цін.
                window (int): Довжина вікна для обчислення історичної волатильності.

            Returns:
                pd.DataFrame: Матриця кореляції волатильностей між активами.
            """
        try:
            self.logger.info(f"Аналіз кросс-активної волатильності для {len(asset_dict)} активів")

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

            self.logger.info("Успішно проаналізовано кросс-активну волатильність")
            return corr_matrix

        except Exception as e:
            self.logger.error(f"Помилка при аналізі кросс-активної волатильності: {e}")
            return pd.DataFrame()

    def extract_seasonality_in_volatility(self, volatility_series, period=7):
        """
            Витягує сезонні патерни з часового ряду волатильності за заданим періодом.

            Підтримувані періоди:
                - 7: день тижня
                - 24: година доби (для внутрішньоденних даних)
                - 30: день місяця
                - 12: місяць року

            Args:
                volatility_series (pd.Series): Часовий ряд волатильності з індексом типу datetime.
                period (int): Період сезонності (7, 24, 30 або 12).

            Returns:
                pd.Series: Середні значення волатильності для кожної категорії в межах періоду (напр., кожен день тижня).
            """
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
            self.logger.error(f"Помилка при вилученні сезонності з волатильності: {e}")
            return pd.Series()

    def analyze_volatility_term_structure(self, symbol, timeframes=['1h', '4h', '1d', '1w']):
        """
            Аналізує структуру термінів волатильності — зміну волатильності в залежності від таймфрейму.

            Здійснює збір даних за різні таймфрейми, обчислює статистики волатильності,
            і додає нормалізовану волатильність відносно денного таймфрейму (1d).

            Args:
                symbol (str): Символ криптовалюти або активу (наприклад, 'BTC').
                timeframes (List[str]): Список таймфреймів (наприклад, ['1h', '1d', '1w']).

            Returns:
                pd.DataFrame: Таблиця зі статистиками волатильності (mean, std, max, min, median) по кожному таймфрейму
                              та колонкою `rel_to_daily` — відносна волатильність до денного.
            """
        try:
            self.logger.info(f"Аналіз терміну структури волатильності для {symbol} на {timeframes}")
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
                        self.logger.error(f"Помилка при обробці таймфрейму {tf}: {e}")
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

            self.logger.info(f"Успішно проаналізовано термін структури волатильності для {symbol}")
            return term_structure

        except Exception as e:
            self.logger.error(f"Помилка при аналізі терміну структури волатильності: {e}")
            return pd.DataFrame()

    def volatility_impulse_response(self, returns, shock_size=3, days=30):
        """
           Оцінює, як змінюється прогнозована волатильність після введення ринкового шоку.

           Створюється GARCH-модель на базових поверненнях, а потім прогноз повторюється після шоку.
           Аналізується різниця прогнозів (імпульсний відгук), та оцінюється час напіврозпаду ефекту.

           Args:
               returns (Union[pd.Series, list]): Часовий ряд логарифмічних або відсоткових повернень.
               shock_size (float): Розмір шоку (в кількостях std).
               days (int): Кількість днів для прогнозу.

           Returns:
               dict: Результат з ключами:
                   - 'impulse' (pd.Series): Різниця між прогнозами волатильності.
                   - 'half_life' (int or None): Дні до падіння ефекту до 50%.
                   - 'shock_size' (float): Розмір шоку.
                   - 'max_effect' (float): Максимальна реакція волатильності.
                   - 'decay_rate' (float): Середній темп зменшення реакції.
           """
        try:
            self.logger.info(f"Обчислення імпульсного відгуку волатильності з розміром шоку {shock_size}")

            # Перетворення вхідних даних на серію, якщо це не серія
            returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns

            # Оптимізація: перетворення в кортеж для кешування (використовуємо хеш-значення)
            returns_key = tuple(returns.values)

            # Отримання базової моделі GARCH
            model, _ = self.fit_garch_model(returns_key, p=1, q=1, model_type='GARCH')

            if model is None:
                self.logger.warning("Не вдалося підігнати GARCH модель для імпульсного відгуку")
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

            self.logger.info(f"Успішно обчислено імпульсний відгук волатильності, час напіврозпаду: {half_life}")

            # Повертаємо імпульсний відгук з додатковими метаданими
            return {
                'impulse': impulse,
                'half_life': half_life,
                'shock_size': shock_size,
                'max_effect': impulse.max(),
                'decay_rate': impulse.diff().mean() if len(impulse) > 1 else 0
            }

        except Exception as e:
            self.logger.error(f"Помилка при обчисленні імпульсного відгуку волатильності: {e}")
            return None

    def prepare_volatility_features_for_ml(self, ohlc_data, window_sizes=[7, 14, 30], include_regimes=True):
        """
        Підготовка ознак волатильності для задач машинного навчання.

        Параметри:
            ohlc_data (pd.DataFrame): OHLC дані з колонками 'open', 'high', 'low', 'close'
            window_sizes (List[int]): Список вікон (у днях) для обчислення функцій волатильності
            include_regimes (bool): Додавати класифікацію режимів волатильності чи ні

        Повертає:
            pd.DataFrame: Таблиця з ознаками, готовими для навчання моделі
        """
        try:
            self.logger.info(f"Підготовка функцій волатільності для ML з вікнами {window_sizes}")

            # Ініціалізація результуючого DataFrame
            features = pd.DataFrame(index=ohlc_data.index)

            # Обчислення прибутків
            returns = ohlc_data['close'].pct_change()
            features['returns'] = returns

            # Безпечне створення ключа кешу
            try:
                # Спробуємо створити хеш з індексу та основних значень
                index_str = str(ohlc_data.index.min()) + str(ohlc_data.index.max()) + str(len(ohlc_data))

                # Використовуємо тільки числові значення для створення хешу
                numeric_sample = []
                for col in ['open', 'high', 'low', 'close']:
                    if col in ohlc_data.columns:
                        col_values = pd.to_numeric(ohlc_data[col], errors='coerce').dropna()
                        if len(col_values) > 0:
                            numeric_sample.extend([float(col_values.iloc[0]), float(col_values.iloc[-1])])

                cache_key = hash(index_str + str(tuple(numeric_sample)))

            except Exception as hash_error:
                # Якщо не вдається створити хеш, використовуємо простий ключ
                self.logger.warning(f"Не вдалося створити хеш для кешу: {hash_error}. Використовуємо простий ключ.")
                cache_key = f"cache_{len(ohlc_data)}_{str(ohlc_data.index.min())}"

            # Перевірка кешу
            if hasattr(self, '_feature_cache') and cache_key in self._feature_cache:
                vol_features = self._feature_cache[cache_key]
            else:
                vol_features = self.feature_engineer.create_volatility_features(ohlc_data)
                if not hasattr(self, '_feature_cache'):
                    self._feature_cache = {}
                self._feature_cache[cache_key] = vol_features

            features = pd.concat([features, vol_features], axis=1)

            # Оптимізація: паралельне обчислення різних вікон волатільності
            if self.use_parallel and len(window_sizes) > 2:
                def compute_window_features(window):
                    try:
                        window_features = {}
                        # Додавання історичної волатільності
                        hist_vol = self.calculate_historical_volatility(ohlc_data['close'], window=window)
                        window_features[f'hist_vol_{window}d'] = hist_vol

                        # Додавання волатільності Паркінсона
                        park_vol = self.calculate_parkinson_volatility(ohlc_data, window=window)
                        window_features[f'park_vol_{window}d'] = park_vol

                        # Додавання відносної волатільності
                        moving_avg_vol = hist_vol.rolling(window=window * 2).mean()
                        # Перевірка на нульові значення перед діленням
                        rel_vol = pd.Series(np.nan, index=hist_vol.index)
                        mask = (moving_avg_vol > 0) & ~moving_avg_vol.isna()
                        rel_vol.loc[mask] = hist_vol.loc[mask] / moving_avg_vol.loc[mask]
                        window_features[f'rel_vol_{window}d'] = rel_vol

                        # Додавання волатільності волатільності
                        vol_of_vol = hist_vol.rolling(window=window).std()
                        window_features[f'vol_of_vol_{window}d'] = vol_of_vol

                        # Тренд волатільності
                        vol_trend = hist_vol.diff(window)
                        window_features[f'vol_trend_{window}d'] = vol_trend

                        # Відношення діапазону High-Low до волатільності
                        hl_range = (ohlc_data['high'] - ohlc_data['low']) / ohlc_data['close']
                        # Перевірка на нульові значення перед діленням
                        hl_range_to_vol = pd.Series(np.nan, index=hist_vol.index)
                        valid_mask = (hist_vol > 0) & ~hist_vol.isna()
                        hl_range_to_vol.loc[valid_mask] = hl_range.loc[valid_mask] / hist_vol.loc[valid_mask]
                        window_features[f'hl_range_to_vol_{window}d'] = hl_range_to_vol

                        return window_features
                    except Exception as e:
                        self.logger.error(f"Помилка при обчисленні функцій для вікна {window}: {e}")
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
                    # Додавання історичної волатільності
                    features[f'hist_vol_{window}d'] = self.calculate_historical_volatility(
                        ohlc_data['close'], window=window)

                    # Додавання волатільності Паркінсона
                    features[f'park_vol_{window}d'] = self.calculate_parkinson_volatility(
                        ohlc_data, window=window)

                    # Додавання відносної волатільності
                    moving_avg_vol = features[f'hist_vol_{window}d'].rolling(window=window * 2).mean()
                    # Перевірка на нульові значення перед діленням
                    rel_vol = pd.Series(np.nan, index=features.index)
                    mask = (moving_avg_vol > 0) & ~moving_avg_vol.isna()
                    rel_vol.loc[mask] = features[f'hist_vol_{window}d'].loc[mask] / moving_avg_vol.loc[mask]
                    features[f'rel_vol_{window}d'] = rel_vol

                    # Додавання волатільності волатільності
                    features[f'vol_of_vol_{window}d'] = features[f'hist_vol_{window}d'].rolling(window=window).std()

                    # Тренд волатільності
                    features[f'vol_trend_{window}d'] = features[f'hist_vol_{window}d'].diff(window)

                    # Відношення діапазону High-Low до волатільності
                    hl_range = (ohlc_data['high'] - ohlc_data['low']) / ohlc_data['close']
                    # Перевірка на нульові значення перед діленням
                    hl_range_to_vol = pd.Series(np.nan, index=features.index)
                    valid_mask = (features[f'hist_vol_{window}d'] > 0) & ~features[f'hist_vol_{window}d'].isna()
                    hl_range_to_vol.loc[valid_mask] = hl_range.loc[valid_mask] / features[f'hist_vol_{window}d'].loc[
                        valid_mask]
                    features[f'hl_range_to_vol_{window}d'] = hl_range_to_vol

            # Додавання ідентифікації режимів, якщо вказано
            if include_regimes:
                # Використання основної волатільності для виявлення режимів
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
                # Конвертуємо в float перед застосуванням log1p
                hist_vol_float = pd.to_numeric(features['hist_vol_14d'], errors='coerce')

                # Обробка нульових або від'ємних значень перед логарифмуванням
                positive_mask = (hist_vol_float > 0) & ~hist_vol_float.isna()
                log_vol = pd.Series(np.nan, index=features.index)
                if positive_mask.any():
                    log_vol.loc[positive_mask] = np.log1p(hist_vol_float.loc[positive_mask])
                features['log_vol'] = log_vol

                # Нормалізована волатільність (z-score) з обробкою нульового стандартного відхилення
                roll_mean = hist_vol_float.rolling(window=30).mean()
                roll_std = hist_vol_float.rolling(window=30).std()

                vol_zscore = pd.Series(np.nan, index=features.index)
                valid_std_mask = (roll_std > 0) & ~roll_std.isna() & ~roll_mean.isna()
                if valid_std_mask.any():
                    vol_zscore.loc[valid_std_mask] = (hist_vol_float.loc[valid_std_mask] - roll_mean.loc[
                        valid_std_mask]) / \
                                                     roll_std.loc[valid_std_mask]
                features['vol_zscore'] = vol_zscore

                # Індикатор аномальної волатільності (>2 стандартних відхилень)
                features['vol_outlier'] = (np.abs(features['vol_zscore']) > 2).astype(int)

            self.logger.info(f"Успішно підготовлено {len(features.columns)} функцій волатільності для ML")
            return features

        except Exception as e:
            self.logger.error(f"Помилка при підготовці функцій волатільності для ML: {e}")
            # Додаємо додаткову інформацію для діагностики
            self.logger.error(f"Тип ohlc_data: {type(ohlc_data)}")
            if hasattr(ohlc_data, 'dtypes'):
                self.logger.error(f"Типи колонок ohlc_data: {ohlc_data.dtypes}")
            return pd.DataFrame(index=ohlc_data.index)

    def save_volatility_analysis_to_db(self, symbol, timeframe, volatility_data, model_data=None, regime_data=None,
                                       features_data=None, cross_asset_data=None):
        """
            Зберігає результати аналізу волатильності до бази даних з підтримкою паралельного збереження.

            Args:
                symbol (str): Тікер фінансового інструменту (наприклад, "AAPL").
                timeframe (str): Таймфрейм аналізу (наприклад, "1d", "1h").
                volatility_data (pd.DataFrame): Дані основних метрик волатильності.
                model_data (dict, optional): Дані моделі волатильності (назва моделі, параметри, статистики тощо).
                regime_data (dict, optional): Дані режимів волатильності (мітки режимів, метод кластеризації, параметри).
                features_data (pd.DataFrame, optional): Функції (features) для машинного навчання, пов’язані з волатильністю.
                cross_asset_data (pd.DataFrame, optional): Дані кореляції крос-активної волатильності.

            Returns:
                dict: Результат операції зі збереження, що містить:
                    - 'overall_success' (bool): Чи успішно було збережено всі дані.
                    - 'detailed_results' (dict): Статуси збереження для кожного типу даних.
                    - 'error' (str, optional): Опис помилки, якщо збереження не вдалося.


            """
        try:
            self.logger.info(f"Збереження аналізу волатильності для {symbol} на таймфреймі {timeframe}")
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
                    )))

                # 3. Збереження даних режиму
                if regime_data is not None:
                    save_tasks.append(('regime', lambda: self.db_manager.save_volatility_regime(
                        symbol=symbol,
                        timeframe=timeframe,
                        regime_labels=regime_data.get('regimes'),
                        method=regime_data.get('method', 'kmeans'),
                        regime_parameters=regime_data.get('params', {})
                    )))

                # 4. Збереження ML функцій
                if features_data is not None and not features_data.empty:
                    save_tasks.append(('features', lambda: self.db_manager.save_volatility_features(
                        symbol=symbol,
                        timeframe=timeframe,
                        features=features_data
                    )))

                # 5. Збереження даних крос-активної волатильності
                if cross_asset_data is not None and not cross_asset_data.empty:
                    save_tasks.append(('cross_asset', lambda: self.db_manager.save_cross_asset_volatility(
                        symbol=symbol,
                        timeframe=timeframe,
                        correlation=cross_asset_data
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
                            self.logger.info(f"Збережено {task_name}: {task_success}")
                        except Exception as e:
                            self.logger.error(f"Помилка при збереженні {task_name}: {e}")
                            results[task_name] = False
                            success = False
            else:
                # Послідовне збереження
                # 1. Збереження основних метрик волатильності
                if volatility_data is not None and not volatility_data.empty:
                    metrics_success = self.db_manager.save_volatility_metrics(
                        symbol=symbol,
                        timeframe=timeframe,
                        metrics=volatility_data
                    )
                    results['metrics'] = metrics_success
                    success = success and metrics_success
                    self.logger.info(f"Збережено метрики волатильності: {metrics_success}")

                # 2. Збереження моделей волатильності (GARCH тощо)
                if model_data is not None:
                    model_success = self.db_manager.save_volatility_model(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_data.get('name', 'garch'),
                        parameters=model_data.get('params', {}),
                        forecast_data=model_data.get('forecast'),
                        model_stats=model_data.get('stats', {})
                    )
                    results['model'] = model_success
                    success = success and model_success
                    self.logger.info(f"Збережено модель волатильності: {model_success}")

                # 3. Збереження даних режиму
                if regime_data is not None:
                    regime_success = self.db_manager.save_volatility_regime(
                        symbol=symbol,
                        timeframe=timeframe,
                        n_regimes=regime_data.get('regimes'),
                        method=regime_data.get('method', 'kmeans'),
                        regime_parameters=regime_data.get('params', {})
                    )
                    results['regime'] = regime_success
                    success = success and regime_success
                    self.logger.info(f"Збережено режими волатильності: {regime_success}")

                # 4. Збереження ML функцій
                if features_data is not None and not features_data.empty:
                    features_success = self.db_manager.save_volatility_features(
                        symbol=symbol,
                        timeframe=timeframe,
                        features=features_data
                    )
                    results['features'] = features_success
                    success = success and features_success
                    self.logger.info(f"Збережено функції волатильності: {features_success}")

                # 5. Збереження даних крос-активної волатильності
                if cross_asset_data is not None and not cross_asset_data.empty:
                    cross_asset_success = self.db_manager.save_cross_asset_volatility(
                        symbol=symbol,
                        timeframe=timeframe,
                        correlation=cross_asset_data
                    )
                    results['cross_asset'] = cross_asset_success
                    success = success and cross_asset_success
                    self.logger.info(f"Збережено крос-активну волатильність: {cross_asset_success}")

            return {'overall_success': success, 'detailed_results': results}

        except Exception as e:
            self.logger.error(f"Помилка при збереженні аналізу волатильності до бази даних: {e}")
            return {'overall_success': False, 'error': str(e)}

    def load_volatility_analysis_from_db(self, symbol, timeframe, start_date=None, end_date=None):
        """
            Завантажує результати аналізу волатильності з бази даних з підтримкою паралельного завантаження.

            Args:
                symbol (str): Тікер фінансового інструменту.
                timeframe (str): Таймфрейм аналізу.
                start_date (str or datetime, optional): Початкова дата діапазону для завантаження даних.
                end_date (str or datetime, optional): Кінцева дата діапазону для завантаження даних.

            Returns:
                dict: Об'єкт із завантаженими даними, що містить:
                    - 'symbol' (str): Тікер інструменту.
                    - 'timeframe' (str): Таймфрейм.
                    - 'volatility_metrics' (pd.DataFrame or None): Дані метрик волатильності.
                    - 'model_data' (dict or None): Дані моделі волатильності.
                    - 'regime_data' (dict or None): Дані режимів волатильності.
                    - 'features_data' (pd.DataFrame or None): ML-функції волатильності.
                    - 'cross_asset_data' (pd.DataFrame or None): Дані крос-активної волатильності.
                    - 'period' (dict): Інформація про діапазон дат ('start_date', 'end_date').
                    - 'summary' (dict, optional): Статистичний підсумок по основній метриці волатильності.
                    - 'error' (str, optional): Опис помилки, якщо завантаження не вдалося.


            """
        try:
            self.logger.info(f"Завантаження аналізу волатильності для {symbol} на таймфреймі {timeframe}")

            # Оптимізація: паралельне завантаження різних типів даних
            results = {}

            if self.use_parallel:
                load_tasks = [
                    ('metrics_data', lambda: self.db_manager.get_volatility_metrics(
                        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                    )),
                    ('model_data', lambda: self.db_manager.get_volatility_model(
                        symbol=symbol, timeframe=timeframe, model_type='garch'
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
                            self.logger.error(f"Помилка при завантаженні {task_name}: {e}")
                            results[task_name] = None
            else:
                # Послідовне завантаження
                # 1. Завантаження метрик волатильності
                results['metrics_data'] = self.db_manager.get_volatility_metrics(
                    symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
                )

                # 2. Завантаження даних моделі волатильності (за замовчуванням GARCH)
                results['model_data'] = self.db_manager.get_volatility_model(
                    symbol=symbol, timeframe=timeframe, model_type='garch'
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

            self.logger.info(f"Успішно завантажено аналіз волатильності для {symbol}")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні аналізу волатильності з бази даних: {e}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e)
            }

    def analyze_crypto_market_conditions(self, symbols=['BTC', 'ETH', 'SOL'], timeframe='1d', window=14):
        """
            Аналізує ринкові умови криптовалют на основі волатильності кількох символів.

            Аргументи:
                symbols (list[str], optional): Список символів криптовалют для аналізу (за замовчуванням ['BTC', 'ETH', 'SOL']).
                timeframe (str, optional): Інтервал таймфрейму для даних (за замовчуванням '1d').
                window (int, optional): Вікно для розрахунку історичної волатильності (за замовчуванням 14).

            Повертає:
                dict: Консолідований звіт про ринкові умови, який містить:
                    - average_market_vol (float): Середня волатильність по ринку.
                    - vol_trend_30d (float or None): Тренд волатильності за останні 30 днів.
                    - vol_dispersion (float): Розкид волатильності між активами.
                    - vol_correlation (float): Середня кореляція волатильностей.
                    - market_phase: Поточна фаза ринку, визначена за волатильністю.
                    - regime_shifts (dict): Інформація про зміни режимів волатильності для кожного символу.
                    - high_vol_assets (list): Активи з високою поточною волатильністю.
                    - low_vol_assets (list): Активи з низькою поточною волатильністю.

            Виключення:
                Логірує помилки та повертає None у випадку виключення.
            """
        try:
            volatilities = {}

            # Get volatility for each symbol
            for symbol in symbols:
                data = self.db_manager.get_klines(f"{symbol}", timeframe=timeframe)
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
            self.logger.error(f"Error analyzing crypto market conditions: {e}")
            return None

    def run_full_volatility_analysis(self, symbol, timeframe='1d', save_to_db=True):
        """
            Запускає повний аналіз волатильності для одного криптосимволу, включаючи розрахунок різних метрик,
            детекцію режимів, сезонності, GARCH-моделювання та збереження результатів.

            Аргументи:
                symbol (str): Символ криптовалюти для аналізу (наприклад, 'BTC').
                timeframe (str, optional): Таймфрейм для даних (за замовчуванням '1d').
                save_to_db (bool, optional): Чи зберігати результати в базу даних (за замовчуванням True).

            Повертає:
                dict: Результати повного аналізу, що містять:
                    - symbol (str): Аналізований символ.
                    - timeframe (str): Таймфрейм аналізу.
                    - volatility_data (list[dict]): Дані волатильності для кожного часу.
                    - latest_volatility (dict): Останні значення різних метрик волатильності.
                    - current_regime (int or None): Поточний режим волатильності.
                    - volatility_clustering (dict): Дані кластери волатильності (автокореляції).
                    - risk_metrics (dict): Ризикові метрики.
                    - seasonality (dict): Сезонні патерни волатильності.
                    - recent_breakouts (int): Кількість останніх проривів волатильності.
                    - garch_forecast: Прогноз волатильності за GARCH моделлю.
                    - impulse_response (list or None): Імпульсна відповідь волатильності.
                    - market_conditions (dict or None): Ринкові умови для контексту.
                    - summary (dict): Ключові підсумкові статистики (тренд, зміни режимів, тощо).
                    - saved_to_db (bool, optional): Підтвердження збереження у базу (якщо save_to_db=True).
                    - error (str, optional): Текст помилки (якщо виникла).
                    - partial_results (list, optional): Часткові результати у разі помилки.

            Виключення:
                Логірує помилки та повертає словник з інформацією про помилку та частковими результатами.
            """
        try:
            self.logger.info(f"Running full volatility analysis for {symbol} on {timeframe} timeframe")

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

            # Create regime data dictionary for database - Convert Series to list
            regime_data = {
                'regimes': vol_df['regime'].tolist(),  # Convert to list
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
            dow_seasonality = self.extract_seasonality_in_volatility(vol_df['hist_vol_14d'], period=7)
            month_seasonality = self.extract_seasonality_in_volatility(vol_df['hist_vol_14d'], period=12)

            # Ensure seasonality data is serializable (convert Series to lists/dicts)
            if isinstance(dow_seasonality, pd.Series):
                seasonality['dow'] = dow_seasonality.to_dict()
            else:
                seasonality['dow'] = dow_seasonality

            if isinstance(month_seasonality, pd.Series):
                seasonality['month'] = month_seasonality.to_dict()
            else:
                seasonality['month'] = month_seasonality

            # Fit GARCH model
            garch_model, garch_forecast = self.fit_garch_model(data['returns'])

            # Extract forecast values if model was successfully fit
            forecast_values = None
            if garch_model is not None and garch_forecast is not None:
                if isinstance(garch_forecast.variance.iloc[-1], pd.Series):
                    forecast_values = garch_forecast.variance.iloc[-1].tolist()
                else:
                    forecast_values = garch_forecast.variance.iloc[-1]

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
            # Convert impulse_response to list if it's a Series
            if isinstance(impulse_response, pd.Series):
                impulse_response = impulse_response.tolist()

            # Prepare features for ML models
            ml_features = self.prepare_volatility_features_for_ml(data)
            # Ensure ml_features is serializable
            if isinstance(ml_features, pd.DataFrame):
                ml_features = ml_features.to_dict(orient='records')

            # Get market-wide conditions for context
            market_conditions = self.analyze_crypto_market_conditions(
                symbols=[symbol, 'BTC', 'ETH', 'SOL'], timeframe=timeframe)
            # Ensure market_conditions values are serializable
            if market_conditions:
                for key, value in market_conditions.items():
                    if isinstance(value, pd.Series):
                        market_conditions[key] = value.tolist()

            # Get cross-asset correlation data
            cross_asset_symbols = ['BTC', 'ETH', 'SOL']
            asset_dict = {}
            for asset in cross_asset_symbols:
                asset_data = self.db_manager.get_klines(f"{asset}", timeframe=timeframe)
                if asset_data is not None and not asset_data.empty:
                    asset_dict[asset] = asset_data['close']

            cross_asset_vol = self.analyze_cross_asset_volatility(asset_dict) if asset_dict else None
            # Ensure cross_asset_vol is serializable
            if isinstance(cross_asset_vol, pd.DataFrame):
                cross_asset_vol = cross_asset_vol.to_dict(orient='records')
            elif isinstance(cross_asset_vol, pd.Series):
                cross_asset_vol = cross_asset_vol.to_dict()

            # Extract scalar values from Series where needed to prevent 'unhashable type: Series' errors
            latest_hist_vol = None
            latest_parkinson = None
            latest_gk = None
            latest_yz = None
            current_regime = None

            if not vol_df.empty:
                latest_hist_vol = vol_df['hist_vol_14d'].iloc[-1]
                latest_hist_vol = latest_hist_vol.item() if hasattr(latest_hist_vol, 'item') else latest_hist_vol

                latest_parkinson = vol_df['parkinson_vol'].iloc[-1]
                latest_parkinson = latest_parkinson.item() if hasattr(latest_parkinson, 'item') else latest_parkinson

                latest_gk = vol_df['gk_vol'].iloc[-1]
                latest_gk = latest_gk.item() if hasattr(latest_gk, 'item') else latest_gk

                latest_yz = vol_df['yz_vol'].iloc[-1]
                latest_yz = latest_yz.item() if hasattr(latest_yz, 'item') else latest_yz

                current_regime = vol_df['regime'].iloc[-1]
                current_regime = current_regime.item() if hasattr(current_regime, 'item') else current_regime

            # Process autocorrelation data to ensure it's serializable
            significant_lags = []
            max_autocorr = 0
            if isinstance(acf_data,
                          pd.DataFrame) and 'autocorrelation' in acf_data.columns and 'lag' in acf_data.columns:
                filtered_lags = acf_data[acf_data['autocorrelation'] > 0.1]['lag']
                significant_lags = filtered_lags.tolist() if isinstance(filtered_lags, pd.Series) else filtered_lags
                max_autocorr = acf_data['autocorrelation'].max()
                max_autocorr = max_autocorr.item() if hasattr(max_autocorr, 'item') else max_autocorr

            # Combine all results
            analysis_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'volatility_data': vol_df.to_dict(orient='records') if isinstance(vol_df, pd.DataFrame) else vol_df,
                'latest_volatility': {
                    'hist_vol_14d': latest_hist_vol,
                    'parkinson': latest_parkinson,
                    'garman_klass': latest_gk,
                    'yang_zhang': latest_yz
                },
                'current_regime': current_regime,
                'volatility_clustering': {
                    'significant_lags': significant_lags,
                    'max_autocorrelation': max_autocorr
                },
                'risk_metrics': risk_metrics,
                'seasonality': seasonality,
                'recent_breakouts': int(vol_df['breakout'].iloc[-30:].sum()) if len(vol_df) >= 30 else 0,
                'garch_forecast': forecast_values,
                'impulse_response': impulse_response,
                'market_conditions': market_conditions
            }

            # Calculate summary stats ensuring scalar values
            avg_volatility = vol_df['hist_vol_14d'].mean()
            avg_volatility = avg_volatility.item() if hasattr(avg_volatility, 'item') else avg_volatility

            volatility_trend = 'increasing' if vol_df['hist_vol_14d'].iloc[-1] > vol_df['hist_vol_14d'].iloc[
                -7] else 'decreasing'

            regime_changes = vol_df['regime'].diff().abs().sum()
            regime_changes = regime_changes.item() if hasattr(regime_changes, 'item') else regime_changes

            vol_of_vol = None
            if len(vol_df) >= 14:
                vol_of_vol = vol_df['hist_vol_14d'].rolling(window=14).std().iloc[-1]
                vol_of_vol = vol_of_vol.item() if hasattr(vol_of_vol, 'item') else vol_of_vol

            current_vs_hist = None
            if not vol_df.empty:
                mean_vol = vol_df['hist_vol_14d'].mean()
                mean_vol = mean_vol.item() if hasattr(mean_vol, 'item') else mean_vol
                if mean_vol != 0:
                    current_vs_hist = latest_hist_vol / mean_vol

            analysis_results['summary'] = {
                'avg_volatility': avg_volatility,
                'volatility_trend': volatility_trend,
                'regime_changes': regime_changes,
                'volatility_of_volatility': vol_of_vol,
                'current_vs_historical': current_vs_hist
            }

            # Save to database if requested
            if save_to_db:
                self.logger.info(f"Saving volatility analysis for {symbol} to database")
                save_success = self.save_volatility_analysis_to_db(
                    symbol=symbol,
                    timeframe=timeframe,
                    volatility_data=vol_df.to_dict(orient='records') if isinstance(vol_df, pd.DataFrame) else vol_df,
                    model_data=model_data,
                    regime_data=regime_data,
                    features_data=ml_features,
                    cross_asset_data=cross_asset_vol
                )
                analysis_results['saved_to_db'] = save_success

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in full volatility analysis for {symbol}: {e}")
            # Return partial results if available
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e),
                'partial_results': locals().get('vol_df', {}).to_dict(orient='records') if isinstance(
                    locals().get('vol_df'), pd.DataFrame) else None
            }


def main():
    # Parameters for test
    symbols = ['BTC', 'SOL', 'ETH']
    timeframe = '1d'

    print(f"=== Starting volatility analysis for {', '.join(symbols)} on timeframe {timeframe} ===")

    # Initialize volatility analyzer
    va = VolatilityAnalysis(use_parallel=True, max_workers=4)

    # Run analysis for each symbol
    results = {}
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        result = va.run_full_volatility_analysis(symbol, timeframe, save_to_db=False)
        results[symbol] = result

        # Check if there are results
        if result.get('error'):
            print(f"[ERROR] Analysis for {symbol} completed with error: {result['error']}")
            continue

        print(f"\n=== Summary for {symbol} ===")
        print(f"Current volatility (hist_vol_14d): {result['latest_volatility'].get('hist_vol_14d')}")
        print(f"Current regime: {result['current_regime']}")
        print(f"Number of breakouts in 30 days: {result['recent_breakouts']}")
        print(f"Sharpe ratio: {result['risk_metrics'].get('sharpe_ratio')}")
        vol_df = pd.DataFrame(result['volatility_data'])
        print(f"Volatility regimes: {set(vol_df['regime'].dropna())}")
        if 'summary' in result:
            print("\n=== Statistics ===")
            for key, value in result['summary'].items():
                print(f"{key}: {value}")

    print("\n=== Analysis completed successfully ===")


if __name__ == "__main__":
    main()