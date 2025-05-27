from typing import Optional, List
from utils.logger import CryptoLogger
import numpy as np
import pandas as pd
import pandas_ta as ta


class TechnicalFeatures:
    def __init__(self):
        self.logger = CryptoLogger('Technical Features')

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
                            sma_result = ta.sma(result_df['close'], length=window)
                            if sma_result is not None and not sma_result.empty:
                                result_df[f'sma_{window}'] = sma_result
                                added_features_count += 1

                # Експоненціальні ковзні середні
                elif indicator == 'ema':
                    for window in [5, 10, 20, 50, 200]:
                        if 'close' in result_df.columns:
                            ema_result = ta.ema(result_df['close'], length=window)
                            if ema_result is not None and not ema_result.empty:
                                result_df[f'ema_{window}'] = ema_result
                                added_features_count += 1

                # Relative Strength Index
                elif indicator == 'rsi':
                    for window in [7, 14, 21]:
                        if 'close' in result_df.columns:
                            rsi_result = ta.rsi(result_df['close'], length=window)
                            if rsi_result is not None and not rsi_result.empty:
                                result_df[f'rsi_{window}'] = rsi_result
                                added_features_count += 1

                # Moving Average Convergence Divergence
                elif indicator == 'macd':
                    if 'close' in result_df.columns:
                        macd_df = ta.macd(result_df['close'], fast=12, slow=26, signal=9)
                        if macd_df is not None and isinstance(macd_df, pd.DataFrame) and not macd_df.empty:
                            # Перевіряємо наявність колонок перед додаванням
                            for col in macd_df.columns:
                                if col not in result_df.columns:
                                    result_df[col] = macd_df[col]
                                    added_features_count += 1
                        else:
                            self.logger.warning("MACD повернув неочікуваний тип даних або порожній результат")

                # Bollinger Bands
                elif indicator == 'bollinger_bands':
                    for window in [20]:
                        if 'close' in result_df.columns:
                            bbands = ta.bbands(result_df['close'], length=window)
                            if bbands is not None and isinstance(bbands, pd.DataFrame) and not bbands.empty:
                                # Перевіряємо наявність колонок перед додаванням
                                for col in bbands.columns:
                                    if col not in result_df.columns:
                                        result_df[col] = bbands[col]
                                        added_features_count += 1

                                # Додаємо розрахунок ширини полос
                                upper_col = f'BBU_{window}_2.0'
                                lower_col = f'BBL_{window}_2.0'
                                middle_col = f'BBM_{window}_2.0'
                                if all(col in result_df.columns for col in [upper_col, lower_col, middle_col]):
                                    # Захист від ділення на нуль
                                    middle_values = result_df[middle_col].replace(0, np.nan)
                                    result_df[f'bb_width_{window}'] = (result_df[upper_col] - result_df[
                                        lower_col]) / middle_values
                                    added_features_count += 1
                            else:
                                self.logger.warning(
                                    "Bollinger Bands повернув неочікуваний тип даних або порожній результат")

                # Stochastic Oscillator
                elif indicator == 'stochastic':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        stoch = ta.stoch(result_df['high'], result_df['low'], result_df['close'], k=14, d=3, smooth_k=3)
                        if stoch is not None and isinstance(stoch, pd.DataFrame) and not stoch.empty:
                            for col in stoch.columns:
                                if col not in result_df.columns:
                                    result_df[col] = stoch[col]
                                    added_features_count += 1
                        else:
                            self.logger.warning("Stochastic повернув неочікуваний тип даних або порожній результат")

                # Average True Range
                elif indicator == 'atr':
                    for window in [14]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            atr_result = ta.atr(result_df['high'], result_df['low'], result_df['close'], length=window)
                            if atr_result is not None and not atr_result.empty:
                                result_df[f'atr_{window}'] = atr_result
                                added_features_count += 1

                # Average Directional Index
                elif indicator == 'adx':
                    for window in [14]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            adx_df = ta.adx(result_df['high'], result_df['low'], result_df['close'], length=window)
                            if adx_df is not None and isinstance(adx_df, pd.DataFrame) and not adx_df.empty:
                                for col in adx_df.columns:
                                    if col not in result_df.columns:
                                        result_df[col] = adx_df[col]
                                        added_features_count += 1
                            else:
                                self.logger.warning("ADX повернув неочікуваний тип даних або порожній результат")

                # On Balance Volume
                elif indicator == 'obv':
                    if all(col in result_df.columns for col in ['close', 'volume']):
                        obv_result = ta.obv(result_df['close'], result_df['volume'])
                        if obv_result is not None and not obv_result.empty:
                            result_df['obv'] = obv_result
                            added_features_count += 1

                # Rate of Change
                elif indicator == 'roc':
                    for window in [5, 10, 20]:
                        if 'close' in result_df.columns:
                            roc_result = ta.roc(result_df['close'], length=window)
                            if roc_result is not None and not roc_result.empty:
                                result_df[f'roc_{window}'] = roc_result
                                added_features_count += 1

                # Commodity Channel Index
                elif indicator == 'cci':
                    for window in [20]:
                        if all(col in result_df.columns for col in ['high', 'low', 'close']):
                            cci_result = ta.cci(result_df['high'], result_df['low'], result_df['close'], length=window)
                            if cci_result is not None and not cci_result.empty:
                                result_df[f'cci_{window}'] = cci_result
                                added_features_count += 1

                # Volume Weighted Average Price (VWAP)
                elif indicator == 'vwap':
                    if all(col in result_df.columns for col in ['high', 'low', 'close', 'volume']):
                        for period in [14, 30]:
                            try:
                                typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
                                vol_tp = result_df['volume'] * typical_price
                                # Додаємо захист від ділення на нуль
                                volume_sum = result_df['volume'].rolling(window=period).sum()
                                volume_sum = volume_sum.replace(0, np.nan)
                                vwap_result = vol_tp.rolling(window=period).sum() / volume_sum
                                result_df[f'vwap_{period}'] = vwap_result
                                added_features_count += 1
                            except Exception as vwap_error:
                                self.logger.warning(f"Помилка при розрахунку VWAP_{period}: {str(vwap_error)}")

                # SuperTrend індикатор
                elif indicator == 'supertrend':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        for period in [10, 20]:
                            try:
                                st_df = ta.supertrend(result_df['high'], result_df['low'], result_df['close'],
                                                      length=period, multiplier=3.0)
                                if st_df is not None and isinstance(st_df, pd.DataFrame) and not st_df.empty:
                                    for col in st_df.columns:
                                        if col not in result_df.columns:
                                            result_df[col] = st_df[col]
                                            added_features_count += 1
                                else:
                                    self.logger.warning(
                                        f"SuperTrend_{period} повернув неочікуваний тип даних або порожній результат")
                            except Exception as st_error:
                                self.logger.warning(f"Помилка при розрахунку SuperTrend_{period}: {str(st_error)}")

                # Полоси Кельтнера
                elif indicator == 'keltner':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        try:
                            kc_df = ta.kc(result_df['high'], result_df['low'], result_df['close'], length=20)
                            if kc_df is not None and isinstance(kc_df, pd.DataFrame) and not kc_df.empty:
                                for col in kc_df.columns:
                                    if col not in result_df.columns:
                                        result_df[col] = kc_df[col]
                                        added_features_count += 1
                            else:
                                self.logger.warning(
                                    "Keltner Channels повернув неочікуваний тип даних або порожній результат")
                        except Exception as kc_error:
                            self.logger.warning(f"Помилка при розрахунку Keltner Channels: {str(kc_error)}")

                # Parabolic SAR
                elif indicator == 'psar':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        try:
                            psar_df = ta.psar(result_df['high'], result_df['low'])
                            if psar_df is not None and isinstance(psar_df, pd.DataFrame) and not psar_df.empty:
                                for col in psar_df.columns:
                                    if col not in result_df.columns:
                                        result_df[col] = psar_df[col]
                                        added_features_count += 1
                            else:
                                self.logger.warning(
                                    "Parabolic SAR повернув неочікуваний тип даних або порожній результат")
                        except Exception as psar_error:
                            self.logger.warning(f"Помилка при розрахунку Parabolic SAR: {str(psar_error)}")

                # Ichimoku Cloud
                elif indicator == 'ichimoku':
                    if all(col in result_df.columns for col in ['high', 'low', 'close']):
                        try:
                            ichimoku_result = ta.ichimoku(result_df['high'], result_df['low'], result_df['close'])

                            if ichimoku_result is not None:
                                if isinstance(ichimoku_result, pd.DataFrame) and not ichimoku_result.empty:
                                    # Якщо повертається DataFrame
                                    for col in ichimoku_result.columns:
                                        if col not in result_df.columns:
                                            result_df[col] = ichimoku_result[col]
                                            added_features_count += 1
                                elif isinstance(ichimoku_result, tuple):
                                    # Якщо повертається tuple з DataFrame'ами або Series
                                    for component in ichimoku_result:
                                        if isinstance(component, (pd.DataFrame, pd.Series)):
                                            if isinstance(component, pd.Series):
                                                component = component.to_frame()
                                            if not component.empty:
                                                for col in component.columns:
                                                    if col not in result_df.columns:
                                                        result_df[col] = component[col]
                                                        added_features_count += 1
                                else:
                                    # Якщо тип невідомий, робимо ручний розрахунок
                                    self.logger.warning(
                                        f"Неочікуваний тип результату Ichimoku: {type(ichimoku_result)}")
                                    self._calculate_ichimoku_manually(result_df)
                                    added_features_count += 5
                            else:
                                # Якщо результат None, робимо ручний розрахунок
                                self.logger.warning("Ichimoku повернув None, використовуємо ручний розрахунок")
                                self._calculate_ichimoku_manually(result_df)
                                added_features_count += 5

                        except Exception as ichimoku_error:
                            self.logger.warning(
                                f"Помилка при розрахунку Ichimoku через pandas_ta: {str(ichimoku_error)}")
                            self._calculate_ichimoku_manually(result_df)
                            added_features_count += 5

                # Додаткові індикатори можна додати тут
                else:
                    self.logger.warning(f"Індикатор {indicator} не підтримується і буде пропущений")

            except Exception as e:
                self.logger.error(f"Помилка при розрахунку індикатора {indicator}: {str(e)}")

        self.logger.info(f"Додано {added_features_count} технічних індикаторів")

        return result_df

    def _calculate_ichimoku_manually(self, result_df: pd.DataFrame) -> None:
        """
        Ручний розрахунок індикатора Ichimoku
        """
        try:
            high = result_df['high']
            low = result_df['low']
            close = result_df['close']

            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = high.rolling(window=9).max()
            period9_low = low.rolling(window=9).min()
            result_df['ISA_9'] = (period9_high + period9_low) / 2

            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = high.rolling(window=26).max()
            period26_low = low.rolling(window=26).min()
            result_df['ISB_26'] = (period26_high + period26_low) / 2

            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            result_df['ITS_9'] = (result_df['ISA_9'] + result_df['ISB_26']) / 2

            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = high.rolling(window=52).max()
            period52_low = low.rolling(window=52).min()
            result_df['IKS_26'] = (period52_high + period52_low) / 2

            # Chikou Span (Lagging Span): Close shifted back 26 periods
            result_df['ICS_26'] = close.shift(-26)

        except Exception as e:
            self.logger.error(f"Помилка при ручному розрахунку Ichimoku: {str(e)}")

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
        avg_body = result_df['candle_body'].rolling(window=window).mean()
        result_df['rel_candle_size'] = np.where(
            (avg_body.notna()) & (avg_body != 0),
            result_df['candle_body'] / avg_body,
            1.0
        )
        added_features_count += 1

        # --- Використання pandas_ta для патернів свічок ---
        try:
            # Викликаємо окремі функції патернів замість cdl_pattern
            patterns_to_add = []

            # Список основних патернів
            pattern_functions = [
                'cdl_doji', 'cdl_hammer', 'cdl_hangingman', 'cdl_shootingstar',
                'cdl_engulfing', 'cdl_harami', 'cdl_piercing', 'cdl_darkcloud',
                'cdl_morningstar', 'cdl_eveningstar', 'cdl_3whitesoldiers', 'cdl_3blackcrows'
            ]

            for pattern_name in pattern_functions:
                try:
                    if hasattr(ta, pattern_name):
                        pattern_func = getattr(ta, pattern_name)
                        pattern_result = pattern_func(
                            open_=result_df['open'],
                            high=result_df['high'],
                            low=result_df['low'],
                            close=result_df['close']
                        )

                        if pattern_result is not None and not pattern_result.empty:
                            # Перетворюємо результат в бінарний формат (0 або 1)
                            pattern_binary = (pattern_result != 0).astype(int)
                            result_df[pattern_name] = pattern_binary
                            added_features_count += 1
                except Exception as e:
                    self.logger.warning(f"Помилка при обчисленні патерну {pattern_name}: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Помилка при використанні pandas_ta patterns: {e}")

        # --- Функція для безпечного перетворення boolean в int ---
        def safe_bool_to_int(series):
            """Безпечно перетворює boolean серію в int, обробляючи NaN значення"""
            try:
                return series.fillna(False).astype(bool).astype(int)
            except Exception:
                return pd.Series(0, index=series.index, dtype=int)

        # --- Патерни з однієї свічки (додаткові) ---
        # 1. Дожі (тіло менше X% від розмаху) - векторизовано
        doji_threshold = 0.1  # 10% від повного розмаху
        doji_condition = result_df['rel_body_size'] < doji_threshold
        result_df['doji'] = safe_bool_to_int(doji_condition)
        added_features_count += 1

        # 2. Молот (векторизовано)
        hammer_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                (result_df['lower_shadow'] > 2 * result_df['candle_body']) &  # довга нижня тінь
                (result_df['upper_shadow'] < 0.2 * result_df['lower_shadow'])  # коротка верхня тінь
        )
        result_df['hammer'] = safe_bool_to_int(hammer_conditions)
        added_features_count += 1

        # 3. Перевернутий молот (векторизовано)
        inv_hammer_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                (result_df['upper_shadow'] > 2 * result_df['candle_body']) &  # довга верхня тінь
                (result_df['lower_shadow'] < 0.2 * result_df['upper_shadow'])  # коротка нижня тінь
        )
        result_df['inverted_hammer'] = safe_bool_to_int(inv_hammer_conditions)
        added_features_count += 1

        # 4. Довгі свічки (векторизовано)
        window = 20
        avg_body = result_df['candle_body'].rolling(window=window).mean()
        long_candle_condition = (result_df['candle_body'] > 1.5 * avg_body) & avg_body.notna()
        result_df['long_candle'] = safe_bool_to_int(long_candle_condition)
        added_features_count += 1

        # 5. Марібозу (векторизовано)
        marubozu_threshold = 0.05  # тіні менше 5% від розмаху
        marubozu_conditions = (
                (result_df['upper_shadow'] < marubozu_threshold * result_df['candle_range']) &
                (result_df['lower_shadow'] < marubozu_threshold * result_df['candle_range']) &
                (result_df['rel_body_size'] > 0.9)  # тіло займає більше 90% розмаху
        )
        result_df['marubozu'] = safe_bool_to_int(marubozu_conditions)
        added_features_count += 1

        # 6. Висока хвиля (High Wave) - новий патерн
        high_wave_conditions = (
                (result_df['rel_body_size'] < 0.3) &  # маленьке тіло
                ((result_df['upper_shadow'] > 3 * result_df['candle_body']) |
                 (result_df['lower_shadow'] > 3 * result_df['candle_body']))  # довга тінь (верхня або нижня)
        )
        result_df['high_wave'] = safe_bool_to_int(high_wave_conditions)
        added_features_count += 1

        # 7. Світлячок (Firefly) - новий патерн
        firefly_conditions = (
                (result_df['rel_body_size'] < 0.25) &  # дуже маленьке тіло
                (result_df['upper_shadow'] < 0.1 * result_df['candle_range']) &  # майже немає верхньої тіні
                (result_df['lower_shadow'] > 2.5 * result_df['candle_body'])  # дуже довга нижня тінь
        )
        result_df['firefly'] = safe_bool_to_int(firefly_conditions)
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
        result_df['bullish_engulfing'] = safe_bool_to_int(bullish_engulfing)
        added_features_count += 1

        # Ведмеже поглинання
        bearish_engulfing = (
                (result_df['candle_direction'].shift(1) == 1) &  # попередня свічка бичача
                (result_df['candle_direction'] == -1) &  # поточна свічка ведмежа
                (result_df['open'] > result_df['close'].shift(1)) &  # відкриття вище закриття попередньої
                (result_df['close'] < result_df['open'].shift(1))  # закриття нижче відкриття попередньої
        )
        result_df['bearish_engulfing'] = safe_bool_to_int(bearish_engulfing)
        added_features_count += 1

        # 2. Ранкова зірка
        morning_star = (
                (result_df['candle_direction'].shift(2) == -1) &  # перша свічка ведмежа
                (result_df['rel_body_size'].shift(1) < 0.3) &  # друга свічка маленька
                (result_df['candle_direction'] == 1) &  # третя свічка бичача
                (result_df['close'] > (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
        )
        result_df['morning_star'] = safe_bool_to_int(morning_star)
        added_features_count += 1

        # 3. Вечірня зірка
        evening_star = (
                (result_df['candle_direction'].shift(2) == 1) &  # перша свічка бичача
                (result_df['rel_body_size'].shift(1) < 0.3) &  # друга свічка маленька
                (result_df['candle_direction'] == -1) &  # третя свічка ведмежа
                (result_df['close'] < (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
        )
        result_df['evening_star'] = safe_bool_to_int(evening_star)
        added_features_count += 1

        # 4. Три білих солдати
        three_white_soldiers = (
                (result_df['candle_direction'].shift(2) == 1) &  # перша свічка бичача
                (result_df['candle_direction'].shift(1) == 1) &  # друга свічка бичача
                (result_df['candle_direction'] == 1) &  # третя свічка бичача
                (result_df['close'].shift(1) > result_df['close'].shift(2)) &  # друга закривається вище першої
                (result_df['close'] > result_df['close'].shift(1))  # третя закривається вище другої
        )
        result_df['three_white_soldiers'] = safe_bool_to_int(three_white_soldiers)
        added_features_count += 1

        # 5. Три чорні ворони
        three_black_crows = (
                (result_df['candle_direction'].shift(2) == -1) &  # перша свічка ведмежа
                (result_df['candle_direction'].shift(1) == -1) &  # друга свічка ведмежа
                (result_df['candle_direction'] == -1) &  # третя свічка ведмежа
                (result_df['close'].shift(1) < result_df['close'].shift(2)) &  # друга закривається нижче першої
                (result_df['close'] < result_df['close'].shift(1))  # третя закривається нижче другої
        )
        result_df['three_black_crows'] = safe_bool_to_int(three_black_crows)
        added_features_count += 1

        # 6. Зірка доджі
        doji_star_condition = result_df['doji'] & (result_df['doji'].shift(1) == 0)
        result_df['doji_star'] = safe_bool_to_int(doji_star_condition)
        added_features_count += 1

        # 7. Пінцет (зверху/знизу)
        pinbar_tolerance = 0.001  # допустима різниця для "однакових" значень

        # Верхній пінцет (однакові максимуми)
        top_pinbar = (
                (np.abs(result_df['high'] - result_df['high'].shift(1)) < pinbar_tolerance * result_df['high']) &
                (result_df['candle_direction'].shift(1) != result_df['candle_direction'])  # різний напрямок свічок
        )
        result_df['top_pinbar'] = safe_bool_to_int(top_pinbar)
        added_features_count += 1

        # Нижній пінцет (однакові мінімуми)
        bottom_pinbar = (
                (np.abs(result_df['low'] - result_df['low'].shift(1)) < pinbar_tolerance * result_df['low']) &
                (result_df['candle_direction'].shift(1) != result_df['candle_direction'])  # різний напрямок свічок
        )
        result_df['bottom_pinbar'] = safe_bool_to_int(bottom_pinbar)
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
        result_df['bullish_harami'] = safe_bool_to_int(bullish_harami)
        added_features_count += 1

        # Ведмеже харамі
        bearish_harami = (
                (result_df['candle_direction'].shift(1) == 1) &  # попередня свічка бичача
                (result_df['candle_direction'] == -1) &  # поточна свічка ведмежа
                (result_df['open'] < result_df['close'].shift(1)) &  # відкриття нижче закриття попередньої
                (result_df['close'] > result_df['open'].shift(1)) &  # закриття вище відкриття попередньої
                (result_df['candle_body'] < result_df['candle_body'].shift(1) * 0.6)  # тіло менше 60% попереднього
        )
        result_df['bearish_harami'] = safe_bool_to_int(bearish_harami)
        added_features_count += 1

        # 9. Висхідний трикутник (новий патерн)
        for period in [5]:
            # Нижні мінімуми зростають, верхні максимуми однакові
            min_low = result_df['low'].rolling(window=period).min()
            max_high = result_df['high'].rolling(window=period).max()

            # Умови для висхідного трикутника
            ascending_triangle = (
                    (result_df['low'] > min_low.shift(1)) &  # поточний мінімум вище попереднього мінімуму
                    (np.abs(result_df['high'] - max_high.shift(1)) < pinbar_tolerance * result_df['high']) &
                    min_low.shift(1).notna() & max_high.shift(1).notna()  # переконуємося, що значення не NaN
            )
            result_df[f'ascending_triangle_{period}'] = safe_bool_to_int(ascending_triangle)
            added_features_count += 1

        # 10. Низхідний трикутник (новий патерн)
        for period in [5]:
            # Верхні максимуми спадають, нижні мінімуми однакові
            min_low = result_df['low'].rolling(window=period).min()
            max_high = result_df['high'].rolling(window=period).max()

            # Умови для низхідного трикутника
            descending_triangle = (
                    (result_df['high'] < max_high.shift(1)) &  # поточний максимум нижче попереднього максимуму
                    (np.abs(result_df['low'] - min_low.shift(1)) < pinbar_tolerance * result_df['low']) &
                    min_low.shift(1).notna() & max_high.shift(1).notna()  # переконуємося, що значення не NaN
            )
            result_df[f'descending_triangle_{period}'] = safe_bool_to_int(descending_triangle)
            added_features_count += 1

        self.logger.info(f"Додано {added_features_count} ознак на основі патернів свічок")

        return result_df

    def create_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Створює специфічні індикатори для криптовалют.

        Args:
            data: DataFrame з OHLCV даними

        Returns:
            DataFrame з доданими специфічними індикаторами
        """
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
                    def find_local_max(x):
                        if len(x) < 3:
                            return 0
                        mid_idx = len(x) // 2
                        return 1 if x.iloc[mid_idx] == x.max() else 0

                    def find_local_min(x):
                        if len(x) < 3:
                            return 0
                        mid_idx = len(x) // 2
                        return 1 if x.iloc[mid_idx] == x.min() else 0

                    result_df[f'local_max_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        find_local_max, raw=False).fillna(0)
                    result_df[f'local_min_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        find_local_min, raw=False).fillna(0)
                    added_features_count += 2

                # 2. Фрактальний індикатор (аналог індикатора Білла Вільямса)
                for window in [5]:  # класично використовується 5
                    def find_fractal_high(x):
                        if len(x) != window:
                            return 0
                        half_window = window // 2
                        center_val = x.iloc[half_window]
                        return 1 if (center_val == x.max() and
                                     center_val != x.iloc[0] and
                                     center_val != x.iloc[-1]) else 0

                    def find_fractal_low(x):
                        if len(x) != window:
                            return 0
                        half_window = window // 2
                        center_val = x.iloc[half_window]
                        return 1 if (center_val == x.min() and
                                     center_val != x.iloc[0] and
                                     center_val != x.iloc[-1]) else 0

                    result_df[f'fractal_high_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        find_fractal_high, raw=False).fillna(0)
                    result_df[f'fractal_low_{window}'] = result_df['close'].rolling(window=window, center=True).apply(
                        find_fractal_low, raw=False).fillna(0)
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
                        price_diff = np.abs(result_df['close'] - fib_level_price)
                        price_tolerance = result_df['close'] * tolerance
                        near_fib = (price_diff < price_tolerance) & fib_level_price.notna()
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

                    def calc_trend_strength(x):
                        if len(x) == 0:
                            return 0
                        return abs(x.sum()) / len(x)

                    result_df[f'trend_strength_{window}'] = direction.rolling(window=window).apply(
                        calc_trend_strength, raw=False).fillna(0)
                    added_features_count += 1

            except Exception as e:
                self.logger.error(f"Помилка при створенні цінових індикаторів: {str(e)}")

        # --- Індикатори на основі об'єму (потребують цінових даних та об'єму) ---
        if 'volume' in available_groups:
            try:
                # 1. Відношення об'єму до цінової зміни (особливо важливо для криптовалют)
                price_change = np.abs(result_df['close'].diff())
                result_df['volume_price_ratio'] = result_df['volume'] / (price_change + 1e-10)
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
                if 'ohlcv' in available_groups:
                    price_range = result_df['high'] - result_df['low']
                    price_range = np.where(price_range == 0, 1e-10, price_range)  # уникаємо ділення на нуль

                    clv = ((result_df['close'] - result_df['low']) - (
                                result_df['high'] - result_df['close'])) / price_range
                    result_df['acc_dist'] = clv * result_df['volume']
                    result_df['acc_dist_cumulative'] = result_df['acc_dist'].cumsum()
                    added_features_count += 2

                # 5. Об'єм відносно середнього (Volume Relative to Average)
                for window in [7, 14, 30]:
                    avg_volume = result_df['volume'].rolling(window=window).mean()
                    result_df[f'volume_rel_avg_{window}'] = result_df['volume'] / (avg_volume + 1e-10)
                    added_features_count += 1

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

        self.logger.info(f"Додано {added_features_count} специфічних індикаторів для криптовалют")

        return result_df