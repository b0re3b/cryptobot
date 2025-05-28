from typing import List
import numpy as np
import pandas as pd

from utils.logger import CryptoLogger


class StatisticalFeatures:
    def __init__(self):
        self.logger = CryptoLogger('Statistical Features')

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

        # Розрахунок волатільності на основі логарифмічної прибутковості
        # ВИПРАВЛЕННЯ: Додаємо перевірку на нульові/від'ємні значення
        price_series = result_df[price_column].copy()

        # Замінюємо нульові або від'ємні значення на попередні валідні значення
        price_series = price_series.mask(price_series <= 0).fillna(method='ffill')
        if price_series.iloc[0] <= 0:
            price_series = price_series.fillna(method='bfill')

        # Перевіряємо, чи залишились валідні значення
        if (price_series <= 0).any():
            self.logger.warning(
                "Виявлено нульові або від'ємні ціни після заповнення. Замінюємо на мінімальне додатне значення.")
            min_positive = price_series[price_series > 0].min()
            if pd.isna(min_positive):
                min_positive = 1.0  # fallback значення
            price_series = price_series.mask(price_series <= 0, min_positive)

        log_returns = np.log(price_series / price_series.shift(1))

        # Видаляємо inf та -inf значення з log_returns
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan)

        # Стандартне відхилення прибутковості для різних вікон
        for window in window_sizes:
            # Класична волатильність як стандартне відхилення прибутковості
            vol_name = f"volatility_{window}d"
            vol_values = log_returns.rolling(window=window, min_periods=max(1, window // 2)).std() * np.sqrt(252)
            result_df[vol_name] = vol_values.fillna(0)
            added_features_count += 1

            # Експоненційно зважена волатільність
            ewm_vol_name = f"ewm_volatility_{window}d"
            ewm_vol_values = log_returns.ewm(span=window, min_periods=max(1, window // 2)).std() * np.sqrt(252)
            result_df[ewm_vol_name] = ewm_vol_values.fillna(0)
            added_features_count += 1

            # Ковзна сума квадратів прибутковостей (для реалізованої волатільності)
            realized_vol_name = f"realized_volatility_{window}d"
            squared_returns = np.square(log_returns).fillna(0)
            realized_vol_values = np.sqrt(
                squared_returns.rolling(window=window, min_periods=max(1, window // 2)).sum() * (252 / window)
            )
            result_df[realized_vol_name] = realized_vol_values.fillna(0)
            added_features_count += 1

            # Відносна волатільність (порівняння з історичною)
            if window > 10:  # Тільки для більших вікон
                long_window = window * 2
                if len(log_returns) > long_window:  # Перевіряємо, що маємо достатньо даних
                    rel_vol_name = f"relative_volatility_{window}d_to_{long_window}d"
                    short_vol = log_returns.rolling(window=window, min_periods=max(1, window // 2)).std()
                    long_vol = log_returns.rolling(window=long_window, min_periods=max(1, long_window // 2)).std()

                    # ВИПРАВЛЕННЯ: Безпечне ділення
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rel_vol_values = short_vol / long_vol
                        rel_vol_values = rel_vol_values.replace([np.inf, -np.inf], np.nan).fillna(1.0)

                    result_df[rel_vol_name] = rel_vol_values
                    added_features_count += 1

        # Додаткові метрики волатільності, якщо є OHLC дані
        if has_ohlc:
            # Перевіряємо на валідність OHLC даних
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                col_data = result_df[col].copy()
                col_data = col_data.mask(col_data <= 0).fillna(method='ffill').fillna(method='bfill')
                if (col_data <= 0).any():
                    min_positive = col_data[col_data > 0].min()
                    if pd.isna(min_positive):
                        min_positive = 1.0
                    col_data = col_data.mask(col_data <= 0, min_positive)
                result_df[col] = col_data

            for window in window_sizes:
                # Garman-Klass волатільність
                gk_vol_name = f"garman_klass_volatility_{window}d"

                # ВИПРАВЛЕННЯ: Безпечний розрахунок логарифмів
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_hl = np.log(result_df['high'] / result_df['low']) ** 2
                    log_co = np.log(result_df['close'] / result_df['open']) ** 2

                # Замінюємо inf значення
                log_hl = pd.Series(log_hl).replace([np.inf, -np.inf], np.nan).fillna(0)
                log_co = pd.Series(log_co).replace([np.inf, -np.inf], np.nan).fillna(0)

                # Формула Garman-Klass: σ² = 0.5 * log(high/low)² - (2*log(2)-1) * log(close/open)²
                gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
                gk_vol_values = np.sqrt(gk_daily.rolling(window=window, min_periods=max(1, window // 2)).mean() * 252)
                result_df[gk_vol_name] = gk_vol_values.fillna(0)
                added_features_count += 1

                # Yang-Zhang волатільність
                if 'open' in result_df.columns:
                    yz_vol_name = f"yang_zhang_volatility_{window}d"

                    # Overnight volatility
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_oc = np.log(result_df['open'] / result_df['close'].shift(1)) ** 2
                    log_oc = pd.Series(log_oc).replace([np.inf, -np.inf], np.nan).fillna(0)

                    overnight_vol = log_oc.rolling(window=window, min_periods=max(1, window // 2)).mean()
                    oc_vol = log_co.rolling(window=window, min_periods=max(1, window // 2)).mean()

                    # Yang-Zhang: використовує overnight та open-to-close волатільність
                    k = 0.34 / (1.34 + (window + 1) / (window - 1))
                    yz_values = overnight_vol + k * oc_vol + (1 - k) * gk_daily.rolling(
                        window=window, min_periods=max(1, window // 2)).mean()

                    result_df[yz_vol_name] = np.sqrt(yz_values * 252).fillna(0)
                    added_features_count += 1

        # Parkinson волатільність (використовує тільки high і low)
        if all(col in result_df.columns for col in ['high', 'low']):
            for window in window_sizes:
                parkinson_name = f"parkinson_volatility_{window}d"

                # ВИПРАВЛЕННЯ: Безпечний розрахунок
                with np.errstate(divide='ignore', invalid='ignore'):
                    parkinson_daily = 1 / (4 * np.log(2)) * np.log(result_df['high'] / result_df['low']) ** 2

                parkinson_daily = pd.Series(parkinson_daily).replace([np.inf, -np.inf], np.nan).fillna(0)
                parkinson_values = np.sqrt(
                    parkinson_daily.rolling(window=window, min_periods=max(1, window // 2)).mean() * 252
                )
                result_df[parkinson_name] = parkinson_values.fillna(0)
                added_features_count += 1

        # Заповнюємо NaN значення
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    self.logger.debug(f"Заповнення NaN значень у стovпці {col}")
                    # Заповнюємо NaN методом forward fill, потім backward fill
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                    # Якщо все ще є NaN, заповнюємо нулем або медіаною
                    if result_df[col].isna().any():
                        median_val = result_df[col].median()
                        if pd.isna(median_val):  # Якщо медіана теж NaN
                            result_df[col] = result_df[col].fillna(0)
                        else:
                            result_df[col] = result_df[col].fillna(median_val)

        self.logger.info(f"Додано {added_features_count} ознак волатільності")
        return result_df

    def create_return_features(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        """
            Створює набір ознак прибутковості на основі цінового стовпця в DataFrame.

            Для кожного періоду з `periods` обчислюються такі ознаки:
              - Відсоткова зміна ціни (return_Xp)
              - Логарифмічна прибутковість (log_return_Xp)
              - Абсолютна зміна ціни (abs_change_Xp)
              - Нормалізована зміна (Z-score) відсоткової зміни (z_score_return_Xp)
              - Напрямок зміни ціни (бінарна ознака: 1, якщо зростання; 0 — інакше) (direction_Xp)

            Параметри:
            ----------
            data : pd.DataFrame
                Вхідні часові ряди з ціновими даними.
            price_column : str, за замовчуванням 'close'
                Назва стовпця з цінами, на основі якого будуть створюватися ознаки.
            periods : List[int], за замовчуванням [1, 3, 5, 7, 14]
                Періоди (в кількості рядків) для розрахунку прибутковості.

            Повертає:
            ----------
            pd.DataFrame
                Копію вхідного DataFrame з доданими ознаками прибутковості.

            Викидає:
            ----------
            ValueError
                Якщо вхідний `price_column` відсутній у даних.

            Особливості:
            -------------
            - Обробка NaN у ціновому стовпці методом прямої та зворотної заповнюваності.
            - Виявлення та корекція нульових і від'ємних цін.
            - Заповнення NaN у створених ознаках нулями.
            - Логування процесу створення ознак.
            """
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

        # ВИПРАВЛЕННЯ: Перевіряємо на нульові/від'ємні значення
        price_series = result_df[price_column].copy()
        if (price_series <= 0).any():
            self.logger.warning("Виявлено нульові або від'ємні ціни. Заміна на валідні значення.")
            price_series = price_series.mask(price_series <= 0).fillna(method='ffill').fillna(method='bfill')

            if (price_series <= 0).any():
                min_positive = price_series[price_series > 0].min()
                if pd.isna(min_positive):
                    min_positive = 1.0
                price_series = price_series.mask(price_series <= 0, min_positive)

            result_df[price_column] = price_series

        # Лічильник доданих ознак
        added_features_count = 0

        # Розрахунок процентної зміни для кожного періоду
        for period in periods:
            # Процентна зміна
            pct_change_name = f"return_{period}p"
            pct_change_values = result_df[price_column].pct_change(periods=period)
            # ВИПРАВЛЕННЯ: Замінюємо inf значення
            pct_change_values = pct_change_values.replace([np.inf, -np.inf], np.nan).fillna(0)
            result_df[pct_change_name] = pct_change_values
            added_features_count += 1

            # Логарифмічна прибутковість
            log_return_name = f"log_return_{period}p"
            with np.errstate(divide='ignore', invalid='ignore'):
                log_return_values = np.log(result_df[price_column] / result_df[price_column].shift(period))
            log_return_values = pd.Series(log_return_values).replace([np.inf, -np.inf], np.nan).fillna(0)
            result_df[log_return_name] = log_return_values
            added_features_count += 1

            # Абсолютна зміна
            abs_change_name = f"abs_change_{period}p"
            result_df[abs_change_name] = result_df[price_column].diff(periods=period).fillna(0)
            added_features_count += 1

            # Нормалізована зміна (Z-score над N періодами)
            z_score_period = min(period * 5, len(result_df))  # беремо більший період для розрахунку статистики
            if z_score_period > period * 2:  # перевіряємо, що маємо достатньо даних для нормалізації
                z_score_name = f"z_score_return_{period}p"
                rolling_mean = pct_change_values.rolling(window=z_score_period,
                                                         min_periods=max(1, z_score_period // 2)).mean()
                rolling_std = pct_change_values.rolling(window=z_score_period,
                                                        min_periods=max(1, z_score_period // 2)).std()

                # ВИПРАВЛЕННЯ: Безпечне ділення для Z-score
                with np.errstate(divide='ignore', invalid='ignore'):
                    z_score_values = (pct_change_values - rolling_mean) / rolling_std
                z_score_values = z_score_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                result_df[z_score_name] = z_score_values
                added_features_count += 1

        # Додаємо ознаку напрямку зміни ціни (бінарна класифікація)
        for period in periods:
            direction_name = f"direction_{period}p"
            # ВИПРАВЛЕННЯ: Використовуємо .iloc або .values для уникнення ambiguous truth value
            return_col = result_df[f"return_{period}p"]
            result_df[direction_name] = (return_col > 0).astype(int)
            added_features_count += 1

        # Заповнюємо NaN значення (особливо на початку часового ряду)
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    # Для ознак напрямку використовуємо 0 (нейтральне значення)
                    if col.startswith("direction_"):
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        # Для інших ознак використовуємо 0
                        result_df[col] = result_df[col].fillna(0)

        self.logger.info(f"Додано {added_features_count} ознак прибутковості")
        return result_df

    def create_volume_features(self, data: pd.DataFrame,
                               window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
            Створює набір ознак на основі об'єму торгів.

            Для кожного вікна з `window_sizes` обчислюються такі ознаки:
              - Ковзне середнє об'єму (volume_ma_X)
              - Відносний об'єм (поточний об'єм / ковзне середнє) (rel_volume_X)
              - Стандартне відхилення об'єму (volume_std_X)
              - Z-score об'єму (volume_zscore_X)
              - Експоненціальне ковзне середнє об'єму (volume_ema_X)
              - Процентна зміна ковзного середнього об'єму (volume_change_X)

            Крім того, обчислюються:
              - Абсолютна зміна об'єму за 1 період (volume_diff_1)
              - Процентна зміна об'єму за 1 період (volume_pct_change_1)
              - Кумулятивний об'єм за день (cumulative_daily_volume) — якщо індекс datetime
              - Індикатор аномального об'єму (volume_anomaly) — коли відносний об'єм 20-періодного середнього > 2

            Параметри:
            ----------
            data : pd.DataFrame
                Вхідний DataFrame з колонкою 'volume'.
            window_sizes : List[int], за замовчуванням [5, 10, 20, 50]
                Список розмірів вікон для розрахунку ковзних статистик.

            Повертає:
            ----------
            pd.DataFrame
                Копія вхідного DataFrame з доданими ознаками об'єму.

            Особливості:
            -------------
            - Якщо відсутній стовпець 'volume', повертає вхідні дані без змін.
            - Обробляє від’ємні та пропущені значення об’єму (замінює на нуль).
            - Заповнює NaN у нових ознаках нулями.
            - Логування процесу створення ознак.
            """
        self.logger.info("Створення ознак на основі об'єму...")

        # Перевірити, що колонка volume існує
        if 'volume' not in data.columns:
            self.logger.warning("Колонка 'volume' відсутня в даних. Ознаки об'єму не будуть створені.")
            return data

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # ВИПРАВЛЕННЯ: Перевіряємо та очищуємо дані volume
        volume_series = result_df['volume'].copy()

        # Замінюємо від'ємні значення та NaN
        if (volume_series < 0).any():
            self.logger.warning("Виявлено від'ємні значення об'єму. Заміна на нульові значення.")
            volume_series = volume_series.mask(volume_series < 0, 0)

        volume_series = volume_series.fillna(0)
        result_df['volume'] = volume_series

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
            # ВИПРАВЛЕННЯ: Безпечне ділення
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_vol_values = result_df['volume'] / result_df[vol_ma_col]
            rel_vol_values = rel_vol_values.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            result_df[rel_vol_col] = rel_vol_values
            added_features_count += 1

            # Стандартне відхилення об'єму
            vol_std_col = f'volume_std_{window}'
            result_df[vol_std_col] = result_df['volume'].rolling(window=window, min_periods=1).std().fillna(0)
            added_features_count += 1

            # Z-score об'єму
            vol_zscore_col = f'volume_zscore_{window}'
            # ВИПРАВЛЕННЯ: Безпечне ділення для Z-score
            vol_mean = result_df[vol_ma_col]
            vol_std = result_df[vol_std_col]

            with np.errstate(divide='ignore', invalid='ignore'):
                vol_zscore_values = (result_df['volume'] - vol_mean) / vol_std
            vol_zscore_values = vol_zscore_values.replace([np.inf, -np.inf], np.nan).fillna(0)
            result_df[vol_zscore_col] = vol_zscore_values
            added_features_count += 1

            # Експоненціальне ковзне середнє об'єму
            vol_ema_col = f'volume_ema_{window}'
            result_df[vol_ema_col] = result_df['volume'].ewm(span=window, min_periods=1).mean()
            added_features_count += 1

            # Зміна об'єму (процентна зміна ковзного середнього)
            vol_change_col = f'volume_change_{window}'
            vol_change_values = result_df[vol_ma_col].pct_change(periods=1) * 100
            vol_change_values = vol_change_values.replace([np.inf, -np.inf], np.nan).fillna(0)
            result_df[vol_change_col] = vol_change_values
            added_features_count += 1

        # Абсолютна зміна об'єму за 1 період
        result_df['volume_diff_1'] = result_df['volume'].diff(periods=1).fillna(0)
        added_features_count += 1

        # Процентна зміна об'єму за 1 період
        vol_pct_change = result_df['volume'].pct_change(periods=1) * 100
        vol_pct_change = vol_pct_change.replace([np.inf, -np.inf], np.nan).fillna(0)
        result_df['volume_pct_change_1'] = vol_pct_change
        added_features_count += 1

        # Обчислення кумулятивного об'єму за день (якщо дані містять внутрішньоденну інформацію)
        if isinstance(result_df.index, pd.DatetimeIndex):
            result_df['cumulative_daily_volume'] = result_df.groupby(result_df.index.date)['volume'].cumsum()
            added_features_count += 1

        # Індикатор аномального об'єму (об'єм, що перевищує середній у 2+ рази)
        # ВИПРАВЛЕННЯ: Використовуємо безпечне порівняння
        if 'rel_volume_20' in result_df.columns:
            anomaly_condition = result_df['rel_volume_20'] > 2.0
            result_df['volume_anomaly'] = anomaly_condition.astype(int)
        else:
            result_df['volume_anomaly'] = 0
        added_features_count += 1

        # Заповнюємо залишкові NaN значення нулями
        for col in result_df.columns:
            if col not in data.columns:  # тільки нові ознаки
                if result_df[col].isna().any():
                    result_df[col] = result_df[col].fillna(0)

        self.logger.info(f"Створено {added_features_count} ознак на основі об'єму.")
        return result_df