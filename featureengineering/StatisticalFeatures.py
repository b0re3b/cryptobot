from typing import List

import numpy as np
import pandas as pd

from utils.logger import CryptoLogger


class StatisticalFeatures():
    def __init__(self):
        self.logger = CryptoLogger('INFO')
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
