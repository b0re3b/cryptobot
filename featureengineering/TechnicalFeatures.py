from typing import Optional, List

import numpy as np
import pandas as pd
import ta


class TechnicalFeatures:
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