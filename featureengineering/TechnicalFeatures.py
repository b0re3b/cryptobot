from typing import Optional, List
from utils.logger import CryptoLogger
import numpy as np
import pandas as pd
import pandas_ta as ta


class TechnicalFeatures:
    def __init__(self):
        self.logger = CryptoLogger('INFO')
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
                'stochastic', 'atr', 'adx', 'obv', 'roc', 'cci',
                'vwap', 'supertrend', 'keltner', 'psar', 'ichimoku'
            ]
            self.logger.info(f"Використовується базовий набір індикаторів: {indicators}")

        # Лічильник доданих ознак
        added_features_count = 0

        # Індикатори на основі бібліотеки pandas_ta
        for indicator in indicators:
            try:
                # Прості ковзні середні
                if indicator == 'sma':
                    for window in [5, 10, 20, 50, 200]:
                        if 'close' in result_df.columns:
                            result_df[f'sma_{window}'] = ta.sma(result_df['close'], length=window)
                            added_features_count += 1

                # Експоненціальні ковзні середні
                elif indicator == 'ema':
                    for window in [5, 10, 20, 50, 200]:
                        if 'close' in result_df.columns:
                            result_df[f'ema_{window}'] = ta.ema(result_df['close'], length=window)
                            added_features_count += 1

                # Relative Strength Index
                elif indicator == 'rsi':
                    for window in [7, 14, 21]:
                        if 'close' in result_df.columns:
                            result_df[f'rsi_{window}'] = ta.rsi(result_df['close'], length=window)
                            added_features_count += 1

                # Moving Average Convergence Divergence
                elif indicator == 'macd':
                    if 'close' in result_df.columns:
                        macd_df = ta.macd(result_df['close'], fast=12, slow=26, signal=9)
                        result_df = pd.concat([result_df, macd_df], axis=1)
                        added_features_count += 3

                # Bollinger Bands
                elif indicator == 'bollinger_bands':
                    for window in [20]:
                        if 'close' in result_df.columns:
                            bbands = ta.bbands(result_df['close'], length=window)
                            result_df = pd.concat([result_df, bbands], axis=1)
                            # Додаємо розрахунок ширини полос
                            result_df[f'bb_width_{window}'] = (result_df[f'BBU_{window}_2.0'] -
                                                               result_df[f'BBL_{window}_2.0']) / result_df[
                                                                  f'BBM_{window}_2.0']
                            added_features_count += 4

                # Stochastic Oscillator
                elif indicator == 'stochastic':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        stoch = ta.stoch(result_df['high'], result_df['low'], result_df['close'], k=14, d=3, smooth_k=3)
                        result_df = pd.concat([result_df, stoch], axis=1)
                        added_features_count += 2

                # Average True Range
                elif indicator == 'atr':
                    for window in [14]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            result_df[f'atr_{window}'] = ta.atr(result_df['high'], result_df['low'],
                                                                result_df['close'], length=window)
                            added_features_count += 1

                # Average Directional Index
                elif indicator == 'adx':
                    for window in [14]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            adx_df = ta.adx(result_df['high'], result_df['low'], result_df['close'], length=window)
                            result_df = pd.concat([result_df, adx_df], axis=1)
                            added_features_count += 3

                # On Balance Volume
                elif indicator == 'obv':
                    if all(col in result_df.columns for col in ['close', 'volume']):
                        result_df['obv'] = ta.obv(result_df['close'], result_df['volume'])
                        added_features_count += 1

                # Rate of Change
                elif indicator == 'roc':
                    for window in [5, 10, 20]:
                        if 'close' in result_df.columns:
                            result_df[f'roc_{window}'] = ta.roc(result_df['close'], length=window)
                            added_features_count += 1

                # Commodity Channel Index
                elif indicator == 'cci':
                    for window in [20]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            result_df[f'cci_{window}'] = ta.cci(result_df['high'], result_df['low'],
                                                                result_df['close'], length=window)
                            added_features_count += 1

                # Volume Weighted Average Price (VWAP)
                elif indicator == 'vwap':
                    if all(col in result_df.columns for col in ['high', 'low', 'close', 'volume']):
                        # pandas_ta не має вбудованого VWAP з параметром довжини, тому робимо власний розрахунок
                        for period in [14, 30]:
                            typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
                            vol_tp = result_df['volume'] * typical_price
                            result_df[f'vwap_{period}'] = vol_tp.rolling(window=period).sum() / result_df[
                                'volume'].rolling(window=period).sum()
                            added_features_count += 1

                # SuperTrend індикатор
                elif indicator == 'supertrend':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        for period in [10, 20]:
                            st_df = ta.supertrend(result_df['high'], result_df['low'], result_df['close'],
                                                  length=period, multiplier=3.0)
                            result_df = pd.concat([result_df, st_df], axis=1)
                            added_features_count += 2  # Додає індикатор і сигнальні лінії

                # Полоси Кельтнера
                elif indicator == 'keltner':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        kc_df = ta.kc(result_df['high'], result_df['low'], result_df['close'], length=20)
                        result_df = pd.concat([result_df, kc_df], axis=1)
                        added_features_count += 3  # Верхня, середня і нижня лінії

                # Parabolic SAR
                elif indicator == 'psar':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        psar_df = ta.psar(result_df['high'], result_df['low'])
                        result_df = pd.concat([result_df, psar_df], axis=1)
                        added_features_count += 2  # PSARl_0.02_0.2 і PSARs_0.02_0.2

                # Ichimoku Cloud
                elif indicator == 'ichimoku':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        ichimoku_df = ta.ichimoku(result_df['high'], result_df['low'], result_df['close'])
                        result_df = pd.concat([result_df, ichimoku_df], axis=1)
                        added_features_count += 5  # Tenkan, Kijun, Senkou A, Senkou B, Chikou

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

    def create_candle_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Створює ознаки на основі патернів свічок з використанням pandas_ta.

        Args:
            data: DataFrame з OHLCV даними

        Returns:
            DataFrame з доданими ознаками свічкових патернів
        """
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

        # --- Базові властивості свічок (векторизовано) ---
        # 1. Тіло свічки (абсолютне)
        result_df['candle_body'] = np.abs(result_df['close'] - result_df['open'])
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
        # Уникаємо ділення на нуль векторизовано
        result_df['rel_body_size'] = np.where(
            result_df['candle_range'] != 0,
            result_df['candle_body'] / result_df['candle_range'],
            0
        )
        added_features_count += 1

        # 6. Напрямок свічки (1 для бичачої, -1 для ведмежої)
        result_df['candle_direction'] = np.sign(result_df['close'] - result_df['open']).fillna(0).astype(int)
        added_features_count += 1

        # 7. Розмір відносно попередніх N свічок (новий індикатор)
        window = 20
        result_df['rel_candle_size'] = result_df['candle_body'] / result_df['candle_body'].rolling(window=window).mean()
        added_features_count += 1

        # --- Використання pandas_ta для патернів свічок ---
        # pandas_ta має вбудовані функції для розпізнавання патернів свічок
        patterns = ta.cdl_pattern(open_=result_df['open'], high=result_df['high'],
                                  low=result_df['low'], close=result_df['close'])
        result_df = pd.concat([result_df, patterns], axis=1)
        added_features_count += len(patterns.columns)

        # --- Патерни з однієї свічки (додаткові) ---
        # 1. Дожі (тіло менше X% від розмаху) - векторизовано
        doji_threshold = 0.1  # 10% від повного розмаху
        result_df['doji'] = (result_df['rel_body_size'] < doji_threshold).astype(int)
        added_features_count += 1

        # 2. Молот (векторизовано)
        hammer_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                (result_df['lower_shadow'] > 2 * result_df['candle_body']) &  # довга нижня тінь
                (result_df['upper_shadow'] < 0.2 * result_df['lower_shadow'])  # коротка верхня тінь
        )
        result_df['hammer'] = hammer_conditions.astype(int)
        added_features_count += 1

        # 3. Перевернутий молот (векторизовано)
        inv_hammer_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                (result_df['upper_shadow'] > 2 * result_df['candle_body']) &  # довга верхня тінь
                (result_df['lower_shadow'] < 0.2 * result_df['upper_shadow'])  # коротка нижня тінь
        )
        result_df['inverted_hammer'] = inv_hammer_conditions.astype(int)
        added_features_count += 1

        # 4. Довгі свічки (векторизовано)
        window = 20
        avg_body = result_df['candle_body'].rolling(window=window).mean()
        result_df['long_candle'] = (result_df['candle_body'] > 1.5 * avg_body).astype(int)
        added_features_count += 1

        # 5. Марібозу (векторизовано)
        marubozu_threshold = 0.05  # тіні менше 5% від розмаху
        marubozu_conditions = (
                (result_df['upper_shadow'] < marubozu_threshold * result_df['candle_range']) &
                (result_df['lower_shadow'] < marubozu_threshold * result_df['candle_range']) &
                (result_df['rel_body_size'] > 0.9)  # тіло займає більше 90% розмаху
        )
        result_df['marubozu'] = marubozu_conditions.astype(int)
        added_features_count += 1

        # 6. Висока хвиля (High Wave) - новий патерн
        high_wave_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                ((result_df['upper_shadow'] > 3 * result_df['candle_body']) |
                 (result_df['lower_shadow'] > 3 * result_df['candle_body']))  # довга тінь (верхня або нижня)
        )
        result_df['high_wave'] = high_wave_conditions.astype(int)
        added_features_count += 1

        # 7. Світлячок (Firefly) - новий патерн
        firefly_conditions = (
                (result_df['rel_body_size'] < 0.25) &  # дуже маленьке тіло
                (result_df['upper_shadow'] < 0.1 * result_df['candle_range']) &  # майже немає верхньої тіні
                (result_df['lower_shadow'] > 2.5 * result_df['candle_body'])  # дуже довга нижня тінь
        )
        result_df['firefly'] = firefly_conditions.astype(int)
        added_features_count += 1

        # --- Патерни з декількох свічок (векторизовано) ---
        # 1. Поглинання
        # Бичаче поглинання
        bullish_engulfing = (
                (result_df['candle_direction'].shift(1) == -1) &  # попередня свічка ведмежа
                (result_df['candle_direction'] == 1) &  # поточна свічка бичача
                (result_df['open'] < result_df['close'].shift(1)) &  # відкриття нижче закриття попередньої
                (result_df['close'] > result_df['open'].shift(1))  # закриття вище відкриття попередньої
        )
        result_df['bullish_engulfing'] = bullish_engulfing.astype(int)
        added_features_count += 1

        # Ведмеже поглинання
        bearish_engulfing = (
                (result_df['candle_direction'].shift(1) == 1) &  # попередня свічка бичача
                (result_df['candle_direction'] == -1) &  # поточна свічка ведмежа
                (result_df['open'] > result_df['close'].shift(1)) &  # відкриття вище закриття попередньої
                (result_df['close'] < result_df['open'].shift(1))  # закриття нижче відкриття попередньої
        )
        result_df['bearish_engulfing'] = bearish_engulfing.astype(int)
        added_features_count += 1

        # 2. Ранкова зірка
        morning_star = (
                (result_df['candle_direction'].shift(2) == -1) &  # перша свічка ведмежа
                (result_df['rel_body_size'].shift(1) < 0.3) &  # друга свічка маленька
                (result_df['candle_direction'] == 1) &  # третя свічка бичача
                (result_df['close'] > (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
        )
        result_df['morning_star'] = morning_star.astype(int)
        added_features_count += 1

        # 3. Вечірня зірка
        evening_star = (
                (result_df['candle_direction'].shift(2) == 1) &  # перша свічка бичача
                (result_df['rel_body_size'].shift(1) < 0.3) &  # друга свічка маленька
                (result_df['candle_direction'] == -1) &  # третя свічка ведмежа
                (result_df['close'] < (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
        )
        result_df['evening_star'] = evening_star.astype(int)
        added_features_count += 1

        # 4. Три білих солдати
        three_white_soldiers = (
                (result_df['candle_direction'].shift(2) == 1) &  # перша свічка бичача
                (result_df['candle_direction'].shift(1) == 1) &  # друга свічка бичача
                (result_df['candle_direction'] == 1) &  # третя свічка бичача
                (result_df['close'].shift(1) > result_df['close'].shift(2)) &  # друга закривається вище першої
                (result_df['close'] > result_df['close'].shift(1))  # третя закривається вище другої
        )
        result_df['three_white_soldiers'] = three_white_soldiers.astype(int)
        added_features_count += 1

        # 5. Три чорні ворони
        three_black_crows = (
                (result_df['candle_direction'].shift(2) == -1) &  # перша свічка ведмежа
                (result_df['candle_direction'].shift(1) == -1) &  # друга свічка ведмежа
                (result_df['candle_direction'] == -1) &  # третя свічка ведмежа
                (result_df['close'].shift(1) < result_df['close'].shift(2)) &  # друга закривається нижче першої
                (result_df['close'] < result_df['close'].shift(1))  # третя закривається нижче другої
        )
        result_df['three_black_crows'] = three_black_crows.astype(int)
        added_features_count += 1

        # 6. Зірка доджі
        result_df['doji_star'] = (result_df['doji'] & (result_df['doji'].shift(1) == 0)).astype(int)
        added_features_count += 1

        # 7. Пінцет (зверху/знизу)
        pinbar_tolerance = 0.001  # допустима різниця для "однакових" значень

        # Верхній пінцет (однакові максимуми)
        top_pinbar = (
                (np.abs(result_df['high'] - result_df['high'].shift(1)) < pinbar_tolerance * result_df['high']) &
                (result_df['candle_direction'].shift(1) != result_df['candle_direction'])  # різний напрямок свічок
        )
        result_df['top_pinbar'] = top_pinbar.astype(int)
        added_features_count += 1

        # Нижній пінцет (однакові мінімуми)
        bottom_pinbar = (
                (np.abs(result_df['low'] - result_df['low'].shift(1)) < pinbar_tolerance * result_df['low']) &
                (result_df['candle_direction'].shift(1) != result_df['candle_direction'])  # різний напрямок свічок
        )
        result_df['bottom_pinbar'] = bottom_pinbar.astype(int)
        added_features_count += 1

        # 8. Харамі (нові патерни)
        # Бичаче харамі
        bullish_harami = (
                (result_df['candle_direction'].shift(1) == -1) &  # попередня свічка ведмежа
                (result_df['candle_direction'] == 1) &  # поточна свічка бичача
                (result_df['open'] > result_df['close'].shift(1)) &  # відкриття вище закриття попередньої
                (result_df['close'] < result_df['open'].shift(1)) &  # закриття нижче відкриття попередньої
                (result_df['candle_body'] < result_df['candle_body'].shift(1) * 0.6)  # тіло менше 60% попереднього
        )
        result_df['bullish_harami'] = bullish_harami.astype(int)
        added_features_count += 1

        # Ведмеже харамі
        bearish_harami = (
                (result_df['candle_direction'].shift(1) == 1) &  # попередня свічка бичача
                (result_df['candle_direction'] == -1) &  # поточна свічка ведмежа
                (result_df['open'] < result_df['close'].shift(1)) &  # відкриття нижче закриття попередньої
                (result_df['close'] > result_df['open'].shift(1)) &  # закриття вище відкриття попередньої
                (result_df['candle_body'] < result_df['candle_body'].shift(1) * 0.6)  # тіло менше 60% попереднього
        )
        result_df['bearish_harami'] = bearish_harami.astype(int)
        added_features_count += 1

        # 9. Висхідний трикутник (новий патерн)
        for period in [5]:
            # Нижні мінімуми зростають, верхні максимуми однакові
            min_low = result_df['low'].rolling(window=period).min()
            max_high = result_df['high'].rolling(window=period).max()

            # Умови для висхідного трикутника
            ascending_triangle = (
                    (result_df['low'] > min_low.shift(1)) &  # поточний мінімум вище попереднього мінімуму
                    (np.abs(result_df['high'] - max_high.shift(1)) < pinbar_tolerance * result_df['high'])
            # максимуми рівні
            )
            result_df[f'ascending_triangle_{period}'] = ascending_triangle.astype(int)
            added_features_count += 1

        # 10. Низхідний трикутник (новий патерн)
        for period in [5]:
            # Верхні максимуми спадають, нижні мінімуми однакові
            min_low = result_df['low'].rolling(window=period).min()
            max_high = result_df['high'].rolling(window=period).max()

            # Умови для низхідного трикутника
            descending_triangle = (
                    (result_df['high'] < max_high.shift(1)) &  # поточний максимум нижче попереднього максимуму
                    (np.abs(result_df['low'] - min_low.shift(1)) < pinbar_tolerance * result_df['low'])
            # мінімуми рівні
            )
            result_df[f'descending_triangle_{period}'] = descending_triangle.astype(int)
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
                self.logger.warning(f"Група індикаторів '{group}' недоступна через відсутні колонки: {missing}")

        # Лічильник доданих ознак
        added_features_count = 0

        # --- Цінові індикатори (потребують тільки цінових даних) ---
        if 'basic' in available_groups:
            try:
                # 1. Хвилі Еліота (спрощений підхід) - виявлення паттернів
                for window in [21, 34]:
                    # Знаходимо локальні максимуми і мінімуми
                    result_df[f'local_max_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        lambda x: 1 if x.argmax() == len(x) // 2 else 0, raw=True)
                    result_df[f'local_min_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        lambda x: 1 if x.argmin() == len(x) // 2 else 0, raw=True)
                    added_features_count += 2

                # 2. Фрактальний індикатор (аналог індикатора Білла Вільямса)
                for window in [5]:  # класично використовується 5
                    half_window = window // 2
                    # Верхні фрактали
                    result_df[f'fractal_high_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        lambda x: 1 if (x[half_window] == max(x)) and (x[half_window] != x[0]) and (
                                    x[half_window] != x[-1]) else 0,
                        raw=True)
                    # Нижні фрактали
                    result_df[f'fractal_low_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        lambda x: 1 if (x[half_window] == min(x)) and (x[half_window] != x[0]) and (
                                    x[half_window] != x[-1]) else 0,
                        raw=True)
                    added_features_count += 2

                # 3. Цикли Фібоначчі (відносні рівні)
                for fib_level in [0.236, 0.382, 0.5, 0.618, 0.786]:
                    # Розрахуємо для різних періодів
                    for window in [55, 89]:  # числа Фібоначчі
                        # Знаходимо мінімум і максимум у вікні
                        roll_max = result_df['close'].rolling(window=window).max()
                        roll_min = result_df['close'].rolling(window=window).min()
                        # Розраховуємо рівень Фібоначчі
                        fib_level_price = roll_min + (roll_max - roll_min) * fib_level
                        # Створюємо індикатор близькості до рівня Фібоначчі
                        tolerance = 0.005  # допустиме відхилення у відсотках
                        near_fib = np.abs(result_df['close'] - fib_level_price) < (result_df['close'] * tolerance)
                        result_df[f'near_fib_{fib_level:.3f}_{window}'] = near_fib.astype(int)
                        added_features_count += 1

                # 4. Середній діапазон денних свічок (власний криптоспецифічний індикатор)
                time_window = 24  # для денних свічок
                result_df['daily_range_ratio'] = result_df['close'].pct_change(periods=time_window).abs()
                added_features_count += 1

                # 5. Тренд-аналізатор (власний індикатор для виявлення сили тренду)
                for window in [7, 14, 30]:
                    # Цей індикатор оцінює послідовність руху ціни у вікні
                    # Рахуємо кількість напрямків руху, що збігаються
                    direction = np.sign(result_df['close'].diff()).fillna(0)
                    result_df[f'trend_strength_{window}'] = direction.rolling(window=window).apply(
                        lambda x: abs(x.sum()) / window, raw=True)
                    added_features_count += 1

            except Exception as e:
                self.logger.error(f"Помилка при створенні цінових індикаторів: {str(e)}")

        # --- Індикатори на основі об'єму (потребують цінових даних та об'єму) ---
        if 'volume' in available_groups:
            try:
                # 1. Відношення об'єму до цінової зміни (особливо важливо для криптовалют)
                result_df['volume_price_ratio'] = result_df['volume'] / (np.abs(result_df['close'].diff()) + 1e-10)
                added_features_count += 1

                # 2. Кумулятивний індекс об'єму (delta)
                close_diff = result_df['close'].diff()
                result_df['volume_delta'] = result_df['volume'] * np.sign(close_diff).fillna(0)
                result_df['cumulative_volume_delta'] = result_df['volume_delta'].cumsum()
                added_features_count += 2

                # 3. Об'ємні аномалії (відхилення від середнього об'єму)
                for window in [20, 50]:
                    mean_volume = result_df['volume'].rolling(window=window).mean()
                    std_volume = result_df['volume'].rolling(window=window).std()
                    result_df[f'volume_zscore_{window}'] = (result_df['volume'] - mean_volume) / (std_volume + 1e-10)
                    added_features_count += 1

                # 4. Накопичення/розподіл (Accumulation/Distribution) індикатор
                clv = ((result_df['close'] - result_df['low']) - (result_df['high'] - result_df['close'])) / (
                            result_df['high'] - result_df['low'] + 1e-10)
                result_df['acc_dist'] = clv * result_df['volume']
                result_df['acc_dist_cumulative'] = result_df['acc_dist'].cumsum()
                added_features_count += 2

                # 5. Об'єм відносно середнього (Volume Relative to Average)
                for window in [7, 14, 30]:
                    result_df[f'volume_rel_avg_{window}'] = result_df['volume'] / result_df['volume'].rolling(
                        window=window).mean()
                    added_features_count += 1

                # 6. Об'ємний профіль (розподіл об'єму за цінами) - спрощений підхід
                for window in [20]:
                    # Зважена ціна по об'єму
                    result_df[f'volume_weighted_price_{window}'] = (result_df['close'] * result_df['volume']).rolling(
                        window=window).sum() / result_df['volume'].rolling(window=window).sum()
                    added_features_count += 1

                # 7. Силові Індекси (ElderRay)
                for period in [13]:
                    ema = result_df['close'].ewm(span=period, adjust=False).mean()
                    result_df[f'elder_bull_power_{period}'] = result_df['high'] - ema
                    result_df[f'elder_bear_power_{period}'] = result_df['low'] - ema
                    added_features_count += 2

            except Exception as e:
                self.logger.error(f"Помилка при створенні індикаторів на основі об'єму: {str(e)}")

        # --- Специфічні для криптовалют індикатори (потребують усіх OHLCV даних) ---
        if 'ohlcv' in available_groups:
            try:
                # 1. Волатильність за годину/добу/тиждень (специфічно для крипторинку, що працює 24/7)
                for periods in [24, 24 * 7]:  # година, доба, тиждень у годинних даних
                    high_period = result_df['high'].rolling(window=periods).max()
                    low_period = result_df['low'].rolling(window=periods).min()
                    result_df[f'volatility_{periods}h'] = (high_period - low_period) / low_period
                    added_features_count += 1

                # 2. Індикатор криптовалютної паніки (аналог "індексу страху і жадібності")
                # Використовує комбінацію волатильності і об'єму
                for window in [24]:
                    volatility = result_df['close'].rolling(window=window).std() / result_df['close'].rolling(
                        window=window).mean()
                    volume_change = result_df['volume'].pct_change(periods=window)
                    # Індекс страху: висока волатильність + різкий зріст об'єму + зниження ціни
                    fear_index = volatility * np.where(
                        (volume_change > 0) & (result_df['close'].pct_change(periods=window) < 0),
                        volume_change + 1, 1)
                    result_df[f'crypto_fear_index_{window}'] = fear_index
                    added_features_count += 1

                # 3. Індикатор різкого руху (для виявлення pump & dump)
                for window in [6, 12, 24]:  # 6 годин, 12 годин, 24 години
                    # Оцінює відносну зміну ціни за короткий період
                    price_change = result_df['close'].pct_change(periods=window).abs()
                    volume_change = result_df['volume'].pct_change(periods=window)
                    # Індикатор = зміна ціни * зміна об'єму
                    result_df[f'pump_dump_indicator_{window}'] = np.where(volume_change > 0,
                                                                          price_change * volume_change, 0)
                    added_features_count += 1

                # 4. Індикатор ф'ючерсного ажіотажу (для криптовалют)
                # На реальних даних тут можна було б використовувати funding rate, але заміняємо на синтетичний розрахунок
                for window in [24, 48]:
                    close_mean = result_df['close'].rolling(window=window).mean()
                    close_std = result_df['close'].rolling(window=window).std()
                    # Індикатор ажіотажу: наскільки поточна ціна відхиляється від середньої в термінах стандартних відхилень
                    result_df[f'futures_heat_{window}'] = (result_df['close'] - close_mean) / (close_std + 1e-10)
                    added_features_count += 1

                # 5. Індикатор відновлення піків/мінімумів
                for percent in [0.05, 0.1]:  # 5% і 10% від попереднього піку/мінімуму
                    for window in [30, 60]:  # час для пошуку піків/мінімумів
                        # Знаходимо піки і мінімуми
                        roll_max = result_df['high'].rolling(window=window).max()
                        roll_min = result_df['low'].rolling(window=window).min()

                        # Перевіряємо, чи наближаємося до піку/мінімуму
                        near_peak = result_df['close'] >= roll_max * (1 - percent)
                        near_bottom = result_df['close'] <= roll_min * (1 + percent)

                        result_df[f'near_peak_{int(percent * 100)}p_{window}'] = near_peak.astype(int)
                        result_df[f'near_bottom_{int(percent * 100)}p_{window}'] = near_bottom.astype(int)
                        added_features_count += 2

                # 6. Індикатор тривалості циклу (специфічний для крипторинку)
                for window in [90]:  # ~3 місяці
                    # Пошук циклів між значними максимумами
                    def find_cycle_position(x):
                        if len(x) < 3:
                            return 0
                        # Знаходимо максимум
                        max_idx = np.argmax(x)
                        # Позиція у циклі (0 = початок, 1 = кінець)
                        if max_idx == 0:
                            return 0  # На початку циклу
                        elif max_idx == len(x) - 1:
                            return 1  # У кінці циклу
                        else:
                            return max_idx / (len(x) - 1)  # Відносна позиція

                    result_df[f'cycle_position_{window}'] = result_df['close'].rolling(window=window).apply(
                        find_cycle_position, raw=False)
                    added_features_count += 1

                # 7. Індикатор консолідації (для виявлення бічних рухів перед сильними рухами)
                for window in [14, 21]:
                    # Обчислюємо відношення діапазону за останні N періодів до середнього періоду
                    high_low_range = result_df['high'].rolling(window=window).max() - result_df['low'].rolling(
                        window=window).min()
                    avg_candle_range = (result_df['high'] - result_df['low']).rolling(window=window).mean()
                    result_df[f'consolidation_{window}'] = avg_candle_range / (high_low_range + 1e-10)
                    added_features_count += 1

            except Exception as e:
                self.logger.error(f"Помилка при створенні специфічних криптовалютних індикаторів: {str(e)}")

        # Заповнюємо пропущені значення
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                # Для заповнення використовуємо forward fill, потім backward fill
                result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                # Якщо все ще є NaN, заповнюємо нулями
                if result_df[col].isna().any():
                    result_df[col] = result_df[col].fillna(0)

        self.logger.info(f"Додано {added_features_count} специфічних індикаторів для криптовалют")

        return result_df