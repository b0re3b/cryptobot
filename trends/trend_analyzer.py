from typing import Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats

from utils.logger import CryptoLogger


class TrendAnalyzer:
    def __init__(self, config=None):
        self.logger = CryptoLogger('trend_analyzer')
        # Ініціалізація конфігурації
        self.config = config or {}

        # Встановлення значень за замовчуванням, якщо не вказано інше
        self.default_window = self.config.get('default_window', 14)
        self.default_threshold = self.config.get('default_threshold', 0.02)
        self.min_points_for_level = self.config.get('min_points_for_level', 3)

    def detect_trend(self, data: pd.DataFrame, window_size: int = 14) -> str:
        """
            Визначає поточний ринковий тренд на основі зваженої системи технічних індикаторів.

            Метод використовує кілька популярних технічних індикаторів (ADX, SMA, MACD, RSI),
            щоб оцінити силу і напрям ринку. Для кожного індикатора враховується його сила
            (величина відхилення або сигнального значення) і ваговий коефіцієнт. Остаточне
            рішення базується на сукупному зваженому голосуванні.

            Args:
                data (pd.DataFrame): OHLCV-дані з обов'язковими стовпцями 'high', 'low' та 'close'.
                window_size (int, optional): Розмір вікна для розрахунку індикаторів. За замовчуванням 14.

            Returns:
                str: Один з варіантів напрямку тренду:
                    - "uptrend" – ринок у висхідному тренді.
                    - "downtrend" – ринок у низхідному тренді.
                    - "sideways" – наявні суперечливі сигнали, тренд невизначений.
                    - "unknown" – недостатньо даних або неможливо визначити тренд.

            Raises:
                ValueError: Якщо вхідний DataFrame не містить обов'язкових стовпців.

            Notes:
                - ADX використовується для визначення сили тренду, з адаптивним масштабуванням ваги.
                - SMA (коротка і довга) аналізується на предмет перетинів та порядку.
                - MACD оцінює імпульс ринку на основі гістограми і лінії сигналу.
                - RSI враховує зони перекупленості / перепроданості з градацією сили сигналу.
                - Метод підтримує fallback-логіку для ручного обрахунку ADX/MACD, якщо бібліотечні методи не спрацювали.
            """
        try:
            self.logger.debug(f"Input DataFrame size: {len(data)}")
            # Перевірка наявності даних
            if data.empty:
                self.logger.warning("Empty DataFrame provided for trend detection")
                return "unknown"

            # Перевірка наявності необхідних стовпців
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"DataFrame must contain '{col}' column")
                    raise ValueError(f"DataFrame must contain '{col}' column")

            df = data.copy()

            # 1. Розрахунок ADX за допомогою pandas-ta
            adx_data = df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=window_size)
            if adx_data is not None:
                df['adx'] = adx_data[f'ADX_{window_size}']
                df['plus_di'] = adx_data[f'DMP_{window_size}']
                df['minus_di'] = adx_data[f'DMN_{window_size}']
            else:
                # Fallback до ручного розрахунку
                df = self._calculate_adx_manual(df, window_size)

            # 2. Розрахунок SMA за допомогою pandas-ta
            df['sma_short'] = df.ta.sma(length=window_size // 2)
            df['sma_long'] = df.ta.sma(length=window_size)

            # 3. Розрахунок MACD за допомогою pandas-ta
            macd_data = df.ta.macd(fast=12, slow=26, signal=9)
            if macd_data is not None:
                df['macd'] = macd_data[f'MACD_{12}_{26}_{9}']
                df['macd_signal'] = macd_data[f'MACDs_{12}_{26}_{9}']
                df['macd_hist'] = macd_data[f'MACDh_{12}_{26}_{9}']
            else:
                # Fallback до ручного розрахунку
                df = self._calculate_macd_manual(df)

            # 4. Розрахунок RSI за допомогою pandas-ta
            df['rsi'] = df.ta.rsi(length=window_size)

            # Беремо останні дані для аналізу
            recent = df.tail(3)  # Використовуємо останні 3 свічки з повними індикаторами

            if len(recent) < 3:
                self.logger.warning("Not enough data points for reliable trend detection")
                return "unknown"

            # ВИПРАВЛЕНО: Використовуємо зважену систему голосування замість простого підрахунку
            trend_signals = {}  # Словник для зберігання сигналів з вагами

            # Вагові коефіцієнти для різних індикаторів
            indicator_weights = {
                'adx': 0.30,  # ADX найважливіший для визначення сили тренду
                'sma': 0.25,  # Ковзні середні добре показують напрямок
                'macd': 0.25,  # MACD показує імпульс
                'rsi': 0.20  # RSI допоміжний індикатор
            }

            # Перевірка ADX (сила тренду)
            if all(col in df.columns for col in ['adx', 'plus_di', 'minus_di']) and not pd.isna(recent['adx'].iloc[-1]):
                last_adx = recent['adx'].iloc[-1]
                last_plus_di = recent['plus_di'].iloc[-1]
                last_minus_di = recent['minus_di'].iloc[-1]

                adx_threshold = self.config.get('adx_threshold', 25)

                # ADX > 25 зазвичай вказує на наявність тренду
                if last_adx >= adx_threshold:
                    if last_plus_di > last_minus_di:
                        trend_signals['adx'] = ('uptrend', indicator_weights['adx'], last_adx / 100.0)
                    else:
                        trend_signals['adx'] = ('downtrend', indicator_weights['adx'], last_adx / 100.0)
                else:
                    # При слабкому ADX, зменшуємо вагу цього сигналу
                    weak_weight = indicator_weights['adx'] * (last_adx / adx_threshold)
                    if last_plus_di > last_minus_di:
                        trend_signals['adx'] = ('uptrend', weak_weight, last_adx / 100.0)
                    else:
                        trend_signals['adx'] = ('downtrend', weak_weight, last_adx / 100.0)

            # Перевірка SMA (ковзні середні)
            if all(col in recent.columns for col in ['sma_short', 'sma_long']) and \
                    not pd.isna(recent['sma_short'].iloc[-1]) and not pd.isna(recent['sma_long'].iloc[-1]):

                close_price = recent['close'].iloc[-1]
                sma_short = recent['sma_short'].iloc[-1]
                sma_long = recent['sma_long'].iloc[-1]

                # Розраховуємо силу сигналу на основі відстані між середніми
                sma_distance = abs(sma_short - sma_long) / sma_long
                strength = min(1.0, sma_distance * 10)  # Нормалізуємо силу сигналу

                if close_price > sma_short > sma_long:
                    trend_signals['sma'] = ('uptrend', indicator_weights['sma'], strength)
                elif close_price < sma_short < sma_long:
                    trend_signals['sma'] = ('downtrend', indicator_weights['sma'], strength)
                else:
                    # Змішаний сигнал, зменшуємо вагу
                    if close_price > sma_long:
                        trend_signals['sma'] = ('uptrend', indicator_weights['sma'] * 0.5, strength * 0.5)
                    else:
                        trend_signals['sma'] = ('downtrend', indicator_weights['sma'] * 0.5, strength * 0.5)

            # Перевірка MACD
            if all(col in recent.columns for col in ['macd', 'macd_signal', 'macd_hist']) and \
                    not pd.isna(recent['macd'].iloc[-1]):

                macd_line = recent['macd'].iloc[-1]
                macd_signal = recent['macd_signal'].iloc[-1]
                macd_hist = recent['macd_hist'].iloc[-1]

                # Розраховуємо силу MACD сигналу
                avg_price = df['close'].mean()
                macd_strength = min(1.0, abs(macd_hist) / avg_price * 1000)  # Нормалізуємо

                if macd_line > macd_signal and macd_hist > 0:
                    trend_signals['macd'] = ('uptrend', indicator_weights['macd'], macd_strength)
                elif macd_line < macd_signal and macd_hist < 0:
                    trend_signals['macd'] = ('downtrend', indicator_weights['macd'], macd_strength)
                else:
                    # Слабкий сигнал MACD
                    weight = indicator_weights['macd'] * 0.3
                    if macd_line > macd_signal:
                        trend_signals['macd'] = ('uptrend', weight, macd_strength * 0.3)
                    else:
                        trend_signals['macd'] = ('downtrend', weight, macd_strength * 0.3)

            # Перевірка RSI
            if 'rsi' in recent.columns and not pd.isna(recent['rsi'].iloc[-1]):
                rsi_value = recent['rsi'].iloc[-1]

                # RSI сигнали з врахуванням зон перекупленості/перепроданості
                if rsi_value > 70:  # Перекупленість, але бичачий тренд
                    trend_signals['rsi'] = ('uptrend', indicator_weights['rsi'] * 0.7, (rsi_value - 50) / 50)
                elif rsi_value < 30:  # Перепроданість, але ведмежий тренд
                    trend_signals['rsi'] = ('downtrend', indicator_weights['rsi'] * 0.7, (50 - rsi_value) / 50)
                elif rsi_value > 55:  # Помірно бичячий
                    trend_signals['rsi'] = ('uptrend', indicator_weights['rsi'] * 0.5, (rsi_value - 50) / 50)
                elif rsi_value < 45:  # Помірно ведмежий
                    trend_signals['rsi'] = ('downtrend', indicator_weights['rsi'] * 0.5, (50 - rsi_value) / 50)
                # Якщо RSI близько до 50, не додаємо сигнал (нейтральна зона)

            # Логуємо сигнали для діагностики
            self.logger.debug(f"Trend signals with weights: {trend_signals}")

            # Якщо сигналів недостатньо, повертаємо unknown
            if len(trend_signals) == 0:
                return "unknown"

            # ВИПРАВЛЕНО: Розраховуємо зважений результат
            uptrend_score = 0.0
            downtrend_score = 0.0

            for indicator, (signal, weight, strength) in trend_signals.items():
                effective_weight = weight * strength  # Вага, скоригована на силу сигналу

                if signal == "uptrend":
                    uptrend_score += effective_weight
                elif signal == "downtrend":
                    downtrend_score += effective_weight

            # Визначаємо мінімальний поріг для впевненості в тренді
            min_confidence_threshold = 0.15  # Мінімальна сума вагових коефіцієнтів
            confidence_difference = 0.05  # Мінімальна різниця між трендами

            # Логуємо розрахунки
            self.logger.debug(f"Trend scores: uptrend={uptrend_score:.3f}, downtrend={downtrend_score:.3f}")

            # Визначаємо остаточний тренд
            if uptrend_score > downtrend_score + confidence_difference and uptrend_score >= min_confidence_threshold:
                return "uptrend"
            elif downtrend_score > uptrend_score + confidence_difference and downtrend_score >= min_confidence_threshold:
                return "downtrend"
            elif uptrend_score >= min_confidence_threshold or downtrend_score >= min_confidence_threshold:
                return "sideways"  # Є сигнали, але вони суперечливі
            else:
                return "unknown"  # Недостатньо впевненості в будь-якому напрямку

        except Exception as e:
            self.logger.error(f"Error in trend detection: {str(e)}")
            return "unknown"

    def _calculate_adx_manual(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
            Ручний розрахунок індикатора ADX (Average Directional Index).

            Цей метод використовується як резервний (fallback) варіант у випадку,
            коли бібліотека `pandas-ta` не може розрахувати ADX. Він виконує
            покроковий обрахунок TR, ATR, +DI, -DI, DX і, зрештою, самого ADX.

            Args:
                df (pd.DataFrame): Вхідний DataFrame з ціновими стовпцями `high`, `low`, `close`.
                period (int, optional): Період для обчислення індикатора. За замовчуванням 14.

            Returns:
                pd.DataFrame: DataFrame з доданими стовпцями:
                    - 'tr': True Range
                    - 'atr': Average True Range
                    - 'plus_dm': +Directional Movement
                    - 'minus_dm': -Directional Movement
                    - 'plus_di': +DI (Positive Directional Index)
                    - 'minus_di': -DI (Negative Directional Index)
                    - 'dx': Directional Index
                    - 'adx': Середній Directional Index (ADX)

            Raises:
                None explicitly, але помилки логуються, і метод повертає вхідний DataFrame з частковими або порожніми результатами.

            Notes:
                - Для розрахунку ATR, +DI та -DI використовується ковзне середнє (`rolling.mean()`).
                - Метод не видаляє проміжні стовпці (наприклад, `high_low`, `high_diff`), які можна згодом очистити.
                - Вхідні дані повинні бути відсортовані за часом.
            """
        try:
            # Розрахунок True Range (TR)
            df['high_low'] = df['high'] - df['low']
            df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
            df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=period).mean()

            # Розрахунок +DM і -DM
            df['high_diff'] = df['high'] - df['high'].shift(1)
            df['low_diff'] = df['low'].shift(1) - df['low']

            df['plus_dm'] = 0.0
            df['minus_dm'] = 0.0

            # Умови для +DM і -DM
            plus_dm_condition = (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0)
            minus_dm_condition = (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0)

            df.loc[plus_dm_condition, 'plus_dm'] = df['high_diff']
            df.loc[minus_dm_condition, 'minus_dm'] = df['low_diff']

            # Розрахунок згладжених +DM і -DM
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])

            # Розрахунок DX і ADX
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            df['adx'] = df['dx'].rolling(window=period).mean()

            return df
        except Exception as e:
            self.logger.error(f"Error in manual ADX calculation: {str(e)}")
            return df

    def _calculate_macd_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """
           Ручний розрахунок індикатора MACD (Moving Average Convergence Divergence).

           Метод використовується як резервний (fallback) варіант для обчислення MACD,
           якщо бібліотека `pandas-ta` недоступна або не працює коректно. Застосовує
           експоненціальні ковзні середні (EMA) з типовими параметрами.

           Args:
               df (pd.DataFrame): Вхідний DataFrame з колонкою `close` (ціна закриття).

           Returns:
               pd.DataFrame: DataFrame з доданими стовпцями:
                   - 'macd': Різниця між EMA(12) та EMA(26)
                   - 'macd_signal': EMA(9) від MACD — сигнальна лінія
                   - 'macd_hist': MACD гістограма (MACD - сигнальна лінія)

           Raises:
               None explicitly, але всі помилки логуються, і метод повертає оригінальний DataFrame з частковими результатами.

           Notes:
               - Значення EMA розраховуються за допомогою `pandas.Series.ewm`.
               - Вхідні дані повинні містити колонку `close` та бути відсортованими за часом.
           """
        try:
            # Розрахунок MACD (Moving Average Convergence Divergence)
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            return df
        except Exception as e:
            self.logger.error(f"Error in manual MACD calculation: {str(e)}")
            return df

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Розрахунок ADX за допомогою pandas-ta"""
        try:
            # Перевірка наявності необхідних стовпців
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"DataFrame must contain '{col}' column for ADX calculation")
                    raise ValueError(f"DataFrame must contain '{col}' column")

            # Копіюємо DataFrame щоб уникнути попереджень
            df = data.copy()

            # Розрахунок ADX з використанням pandas-ta
            adx_result = df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=period)

            if adx_result is not None:
                # Додаємо результати до вхідного DataFrame
                df['adx'] = adx_result[f'ADX_{period}']
                df['plus_di'] = adx_result[f'DMP_{period}']  # +DI (Positive Directional Indicator)
                df['minus_di'] = adx_result[f'DMN_{period}']  # -DI (Negative Directional Indicator)
            else:
                # Fallback до ручного розрахунку
                df = self._calculate_adx_manual(df, period)

            # Логуємо успішне виконання
            self.logger.debug(f"ADX calculated successfully for {len(df)} data points with period {period}")

            return df

        except Exception as e:
            self.logger.error(f"Error in ADX calculation: {str(e)}")
            # Повертаємо вхідні дані без змін у випадку помилки
            return data

    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
           Обчислює силу ринкового тренду на основі комбінації технічних індикаторів.

           Метод інтегрує кілька технічних індикаторів (ADX, регресійний нахил, Aroon, MACD, RSI, свічкові патерни,
           волатильність і об'єм) у зважену метрику сили тренду в діапазоні [0.0, 1.0], де 1.0 — сильний тренд.

           Args:
               data (pd.DataFrame): Історичні цінові дані. Обов'язкові колонки:
                   - 'close': ціна закриття (обов'язкова)
                   - 'high', 'low': для ADX, Aroon, волатильності (опціонально)
                   - 'open': для аналізу свічкових патернів (опціонально)
                   - 'volume': для аналізу об'єму (опціонально)

           Returns:
               float: Значення сили тренду в діапазоні [0.0, 1.0].

           Notes:
               - Якщо певні індикатори недоступні через відсутність колонок, використовується дефолтне або нейтральне значення.
               - Використовуються методи з `pandas-ta` для розрахунку технічних індикаторів.
               - Обробляє всі виключення всередині, логує помилки через `self.logger`, і повертає 0.0 при критичних збоях.

           Components and Weights:
               - ADX: 0.25 — сила тренду
               - R² (регресія): 0.15 — трендовість лінії
               - Aroon oscillator: 0.15 — тривалість і сила
               - MACD: 0.15 — імпульс
               - RSI: 0.10 — перекупленість/перепроданість
               - Candle patterns: 0.05 — підтвердження напрямку
               - Volatility (ATR): 0.10 — згладжуючий зворотний фактор
               - Volume: 0.05 — підсилення активністю


           """
        try:
            # Перевірка наявності достатньої кількості даних
            if data.empty or len(data) < 20:
                self.logger.warning("Insufficient data for trend strength calculation (minimum 20 points required)")
                return 0.0

            # Перевіряємо наявність необхідних колонок
            if 'close' not in data.columns:
                self.logger.error("Data must contain 'close' column for trend strength calculation")
                raise ValueError("Data must contain 'close' column")

            # Створюємо копію DataFrame для уникнення попереджень
            df = data.copy()

            # 1. Розрахунок ADX за допомогою pandas-ta
            if all(col in df.columns for col in ['high', 'low', 'close']):
                adx_data = df.ta.adx(length=14)
                if adx_data is not None and f'ADX_14' in adx_data.columns:
                    adx_value = min(1.0, adx_data[f'ADX_14'].iloc[-1] / 100.0) if not pd.isna(
                        adx_data[f'ADX_14'].iloc[-1]) else 0.5
                else:
                    adx_value = 0.5
            else:
                adx_value = 0.5
                self.logger.warning("No high/low data for ADX calculation, using default value")

            # 2. Розрахунок лінійної регресії для визначення напрямку і сили тренду
            x = np.arange(len(df))
            y = df['close'].values
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            r_squared = r_value ** 2  # Коефіцієнт детермінації (0-1)
            trend_direction = np.sign(slope)  # 1 для висхідного, -1 для низхідного

            # 3. Розрахунок Aroon-індикатору за допомогою pandas-ta
            aroon_strength = 0.5  # Значення за замовчуванням

            if all(col in df.columns for col in ['high', 'low']):
                aroon_data = df.ta.aroon(length=14)
                if aroon_data is not None and f'AROONU_14' in aroon_data.columns and f'AROOND_14' in aroon_data.columns:
                    aroon_up = aroon_data[f'AROONU_14'].iloc[-1]
                    aroon_down = aroon_data[f'AROOND_14'].iloc[-1]
                    if not (pd.isna(aroon_up) or pd.isna(aroon_down)):
                        aroon_oscillator = (aroon_up - aroon_down) / 100.0  # від -1 до 1
                        aroon_strength = abs(aroon_oscillator)  # сила тренду по Aroon (0-1)

            # 4. Розрахунок MACD за допомогою pandas-ta
            macd_data = df.ta.macd(fast=12, slow=26, signal=9)
            if macd_data is not None and f'MACD_12_26_9' in macd_data.columns:
                macd_line = macd_data[f'MACD_12_26_9'].iloc[-1]
                if not pd.isna(macd_line):
                    # Нормалізований MACD (відносно ціни)
                    avg_price = df['close'].mean()
                    macd_norm = abs(macd_line) / avg_price
                    macd_strength = min(1.0, macd_norm * 100)  # Обмежуємо до 1.0
                else:
                    macd_strength = 0.0
            else:
                macd_strength = 0.0

            # 5. ВИПРАВЛЕНО: Розрахунок RSI за допомогою pandas-ta (замість некоректної реалізації)
            rsi_data = df.ta.rsi(length=14)
            if rsi_data is not None:
                rsi_value = rsi_data.iloc[-1]
                if not pd.isna(rsi_value):
                    # Перетворюємо RSI на індикатор сили тренду (екстремальні значення = сильний тренд)
                    if rsi_value >= 70 or rsi_value <= 30:
                        # Екстремальні значення RSI вказують на сильний тренд
                        rsi_strength = (abs(rsi_value - 50) - 20) / 30  # Перетворюємо на діапазон 0-1
                        rsi_strength = max(0, min(1, rsi_strength))  # Обмежуємо діапазоном 0-1
                    else:
                        # Значення RSI ближче до 50 вказують на слабший тренд
                        rsi_strength = 0.2  # Базове значення для середнього діапазону
                else:
                    rsi_strength = 0.2
            else:
                rsi_strength = 0.2

            # 6. Аналіз патернів свічок
            candle_pattern_strength = 0.0
            if all(col in df.columns for col in ['open', 'high', 'low']):
                # Визначаємо останні 3 свічки
                recent = df.tail(3)
                # Рахуємо тіла свічок (абсолютна різниця між відкриттям і закриттям)
                bodies = abs(recent['close'] - recent['open'])
                if len(bodies) > 0:
                    avg_body = bodies.mean()

                    # Перевіряємо, чи мають свічки послідовний напрямок і великі тіла
                    close_changes = recent['close'] - recent['open']
                    # Перевіряємо чи всі зміни мають той самий знак що і загальний тренд
                    consistent_direction = all(
                        np.sign(close_changes) == trend_direction) if trend_direction != 0 else False
                    large_bodies = all(bodies > avg_body * 0.8) if avg_body > 0 else False

                    if consistent_direction and large_bodies:
                        candle_pattern_strength = 0.2

            # 7. Розрахунок волатільності за допомогою pandas-ta
            atr_data = df.ta.atr(length=14)
            if atr_data is not None and not pd.isna(atr_data.iloc[-1]):
                # Нормалізована волатільність
                normalized_volatility = atr_data.iloc[-1] / df['close'].iloc[-1]
                volatility_factor = 1.0 - min(1.0,
                                              normalized_volatility * 20)  # Висока волатільність зменшує силу тренду
            else:
                volatility_factor = 0.5

            # 8. Аналіз об'єму (якщо доступний)
            volume_factor = 0.5  # Нейтральне значення за замовчуванням
            if 'volume' in df.columns and len(df) >= 20:
                # ВИПРАВЛЕНО: Розрахунок SMA для об'єму за допомогою pandas-ta на DataFrame
                volume_sma = df.ta.sma(close=df['volume'], length=20)
                if volume_sma is not None and not pd.isna(volume_sma.iloc[-1]):
                    recent_volume = df['volume'].tail(5).mean()
                    historical_volume = volume_sma.iloc[-1]

                    # Високий об'єм підсилює тренд
                    if historical_volume > 0:
                        volume_ratio = recent_volume / historical_volume
                        volume_factor = min(1.0, volume_ratio / 2)  # Нормалізуємо (max 1.0)

            # Розраховуємо зважене значення сили тренду
            weights = {
                'adx': 0.25,  # ADX (основний індикатор сили тренду)
                'r_squared': 0.15,  # Сила лінійної регресії
                'aroon': 0.15,  # Aroon індикатор
                'macd': 0.15,  # MACD імпульс
                'rsi': 0.10,  # RSI екстремуми
                'candle': 0.05,  # Патерни свічок
                'volatility': 0.10,  # Фактор волатільності (зворотний)
                'volume': 0.05  # Фактор об'єму
            }

            # Обчислюємо загальну силу тренду як зважену суму компонентів
            trend_strength = (
                    weights['adx'] * adx_value +
                    weights['r_squared'] * r_squared +
                    weights['aroon'] * aroon_strength +
                    weights['macd'] * macd_strength +
                    weights['rsi'] * rsi_strength +
                    weights['candle'] * candle_pattern_strength +
                    weights['volatility'] * volatility_factor +
                    weights['volume'] * volume_factor
            )

            # Нормалізуємо до діапазону від 0 до 1
            trend_strength = max(0.0, min(1.0, trend_strength))

            # Логуємо компоненти для діагностики
            self.logger.debug(
                f"Trend strength calculation: "
                f"ADX={adx_value:.2f}, R²={r_squared:.2f}, Aroon={aroon_strength:.2f}, "
                f"MACD={macd_strength:.2f}, RSI={rsi_strength:.2f}, "
                f"Volatility={volatility_factor:.2f}, Volume={volume_factor:.2f}, "
                f"Final={trend_strength:.2f}"
            )

            return float(trend_strength)

        except Exception as e:
            self.logger.error(f"Error in trend strength calculation: {str(e)}")
            return 0.0
    def calculate_trend_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
                Обчислює набір метрик, що характеризують трендову поведінку часових рядів ціни.

                Метод аналізує колонку 'close' в переданому DataFrame, а також 'volume', якщо вона присутня,
                і розраховує різноманітні статистичні та технічні характеристики, які можуть бути корисними
                для аналізу трендів, виявлення напрямку руху ціни та сили тренду.

                Очікує, що дані мають часовий індекс та відсортовані за зростанням дати.

                Повертає словник з наступними метриками:

                - speed_5, speed_10, speed_20: середня абсолютна зміна ціни за останні 5, 10 та 20 періодів
                - speed_pct_5, speed_pct_10, speed_pct_20: середня відносна зміна ціни (%) за останні 5, 10, 20 періодів
                - acceleration_5, acceleration_10, acceleration_20: середнє прискорення ціни (зміна швидкості)
                - volatility_5, volatility_10, volatility_20: стандартне відхилення відносної зміни ціни
                - trend_strength: сила тренду на основі RSI-подібного розрахунку
                - trend_linearity: кореляція між ціною та часом (показник лінійності тренду)
                - trend_slope, trend_intercept: нахил та перетин лінійної регресії останніх 20 точок
                - r_squared: коефіцієнт детермінації R² для регресійної моделі
                - trend_efficiency: співвідношення чистої зміни до загальної амплітуди змін за 20 періодів
                - volume_factor: співвідношення останнього об'єму торгів до середнього (за наявності 'volume')
                - above_sma_50, above_sma_200: чи знаходиться ціна вище ковзаючих середніх за 50 і 200 періодів
                - composite_strength: композитний індикатор сили тренду, що враховує волатильність і лінійність

                Параметри:
                ----------
                data : pd.DataFrame
                    Часовий ряд з колонкою 'close' та (необов’язково) 'volume'. Повинен мати datetime-індекс.

                Повертає:
                --------
                Dict[str, float]
                    Словник із розрахованими метриками тренду.

                Винятки:
                --------
                ValueError:
                    Якщо у вхідному DataFrame відсутня колонка 'close'.
                """
        # Перевірка наявності необхідних даних
        if 'close' not in data.columns:
            raise ValueError("DataFrame повинен містити стовпець 'close'")

        # Копіюємо дані для запобігання зміни оригінального DataFrame
        df = data.copy()

        # Переконуємось, що дані відсортовані за датою
        if hasattr(df.index, 'is_monotonic_increasing') and not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

        # Розрахунок базових метрик

        # 1. Швидкість тренду (середня зміна ціни за період)
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100

        # Швидкість за різні періоди
        speed_5 = df['price_change'].tail(5).mean()
        speed_10 = df['price_change'].tail(10).mean()
        speed_20 = df['price_change'].tail(20).mean()

        # Відносна швидкість у відсотках
        speed_pct_5 = df['price_change_pct'].tail(5).mean()
        speed_pct_10 = df['price_change_pct'].tail(10).mean()
        speed_pct_20 = df['price_change_pct'].tail(20).mean()

        # 2. Прискорення тренду (зміна швидкості)
        df['acceleration'] = df['price_change'].diff()
        df['acceleration_pct'] = df['price_change_pct'].diff()

        # Прискорення за різні періоди
        acceleration_5 = df['acceleration'].tail(5).mean()
        acceleration_10 = df['acceleration'].tail(10).mean()
        acceleration_20 = df['acceleration'].tail(20).mean()

        # 3. Волатильність (стандартне відхилення змін ціни)
        volatility_5 = df['price_change_pct'].tail(5).std()
        volatility_10 = df['price_change_pct'].tail(10).std()
        volatility_20 = df['price_change_pct'].tail(20).std()

        # 4. Відносна сила тренду (RSI-подібний підхід)
        df['gain'] = df['price_change'].clip(lower=0)
        df['loss'] = -df['price_change'].clip(upper=0)

        avg_gain_14 = df['gain'].tail(14).mean()
        avg_loss_14 = df['loss'].tail(14).mean()

        rs = avg_gain_14 / avg_loss_14 if avg_loss_14 != 0 else 100
        trend_strength = 100 - (100 / (1 + rs))

        # 5. Кореляція з часом (показує лінійність тренду)
        df['time_index'] = np.arange(len(df))
        trend_linearity = df['close'].corr(df['time_index'])

        # 6. Тренд лінійної регресії
        x = np.arange(len(df)).reshape(-1, 1)
        y = df['close'].values

        # Використовуємо останні 20 періодів для лінійної регресії
        if len(df) >= 20:
            x_recent = x[-20:]
            y_recent = y[-20:]

            # Лінійна регресія для визначення нахилу тренду
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(x_recent, y_recent)
                trend_slope = model.coef_[0]
                trend_intercept = model.intercept_
            except ImportError:
                # Якщо sklearn не доступний, використовуємо простіший підхід
                n = len(x_recent)
                x_mean = np.mean(x_recent)
                y_mean = np.mean(y_recent)
                xy_sum = np.sum(x_recent * y_recent) - n * x_mean * y_mean
                x_squared_sum = np.sum(x_recent ** 2) - n * x_mean ** 2
                trend_slope = xy_sum / x_squared_sum if x_squared_sum != 0 else 0
                trend_intercept = y_mean - trend_slope * x_mean
        else:
            trend_slope = np.nan
            trend_intercept = np.nan

        # 7. R-squared (показує якість лінійної моделі)
        if len(df) >= 20:
            try:
                from sklearn.metrics import r2_score
                y_pred = trend_slope * x_recent + trend_intercept
                r_squared = r2_score(y_recent, y_pred)
            except ImportError:
                # Спрощений розрахунок R-squared
                y_pred = trend_slope * x_recent + trend_intercept
                ss_total = np.sum((y_recent - np.mean(y_recent)) ** 2)
                ss_residual = np.sum((y_recent - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        else:
            r_squared = np.nan

        # 8. Індекс ефективності тренду (Trend Efficiency Index)
        price_moves = df['price_change'].abs().tail(20).sum()
        price_net_change = abs(df['close'].iloc[-1] - df['close'].iloc[-21]) if len(df) >= 21 else abs(
            df['close'].iloc[-1] - df['close'].iloc[0])
        trend_efficiency = price_net_change / price_moves if price_moves != 0 else 0

        # 9. Відносний об'єм (для підтвердження сили тренду)
        if 'volume' in df.columns:
            avg_volume = df['volume'].mean()
            recent_volume = df['volume'].tail(5).mean()
            volume_factor = recent_volume / avg_volume if avg_volume != 0 else 1
        else:
            volume_factor = None

        # Формуємо словник з метриками
        metrics = {
            'speed_5': float(speed_5),
            'speed_10': float(speed_10),
            'speed_20': float(speed_20),
            'speed_pct_5': float(speed_pct_5),
            'speed_pct_10': float(speed_pct_10),
            'speed_pct_20': float(speed_pct_20),
            'acceleration_5': float(acceleration_5),
            'acceleration_10': float(acceleration_10),
            'acceleration_20': float(acceleration_20),
            'volatility_5': float(volatility_5),
            'volatility_10': float(volatility_10),
            'volatility_20': float(volatility_20),
            'trend_strength': float(trend_strength),
            'trend_linearity': float(trend_linearity),
            'trend_slope': float(trend_slope) if not np.isnan(trend_slope) else None,
            'trend_intercept': float(trend_intercept) if not np.isnan(trend_intercept) else None,
            'r_squared': float(r_squared) if not np.isnan(r_squared) else None,
            'trend_efficiency': float(trend_efficiency),
        }

        # Додаємо метрику пов'язану з об'ємом, якщо вона доступна
        if volume_factor is not None:
            metrics['volume_factor'] = float(volume_factor)

        # Додаткові метрики для визначення напрямку тренду
        last_close = df['close'].iloc[-1]
        price_sma_50 = df['close'].tail(50).mean() if len(df) >= 50 else None
        price_sma_200 = df['close'].tail(200).mean() if len(df) >= 200 else None

        if price_sma_50 is not None:
            metrics['above_sma_50'] = float(last_close > price_sma_50)

        if price_sma_200 is not None:
            metrics['above_sma_200'] = float(last_close > price_sma_200)

        # Додаємо індикатор сили тренду на основі комбінації метрик
        # Простий композитний індикатор на основі швидкості та лінійності
        composite_strength = (speed_pct_20 / volatility_20 if volatility_20 != 0 else 0) * trend_linearity
        metrics['composite_strength'] = float(composite_strength)

        return metrics
    def estimate_trend_duration(self, data: pd.DataFrame) -> Dict[str, int]:
        """
                Оцінює тривалість поточного та попередніх трендів на основі ковзних середніх.

                Метод використовує коротко- та довгострокові ковзні середні (SMA20 і SMA50) для визначення
                напрямку тренду (вгору, вниз або флет) у кожному часовому періоді. Далі визначається тривалість
                поточного тренду, середня тривалість трендів, найтриваліший тренд, а також дата початку поточного тренду.

                Для класифікації тренду:
                - 'uptrend': якщо SMA20 > SMA50
                - 'downtrend': якщо SMA20 < SMA50
                - 'sideways': якщо SMA20 == SMA50

                Параметри:
                ----------
                data : pd.DataFrame
                    Часовий ряд з колонкою 'close' і datetime-індексом. Дані повинні бути відсортовані за зростанням часу.

                Повертає:
                --------
                Dict[str, int]
                    Словник з наступними метриками:
                    - current_trend: напрямок поточного тренду ('uptrend', 'downtrend', 'sideways')
                    - current_trend_duration: тривалість поточного тренду в періодах
                    - longest_trend_duration: максимальна тривалість будь-якого тренду
                    - average_trend_duration: середня тривалість трендів
                    - trend_start_date: дата початку поточного тренду
                    - total_periods_analyzed: загальна кількість проаналізованих періодів

                Винятки:
                --------
                ValueError:
                    Якщо у вхідному DataFrame відсутня колонка 'close'.


                """
        # Перевірка наявності необхідних даних
        if 'close' not in data.columns:
            raise ValueError("DataFrame повинен містити стовпець 'close'")

        # Використовуємо pandas-ta для розрахунку ковзних середніх
        data = data.copy()  # Працюємо з копією, щоб не змінювати оригінальний датафрейм
        data['sma20'] = ta.sma(data['close'], length=20)
        data['sma50'] = ta.sma(data['close'], length=50)

        # Визначаємо напрямок за ковзними середніми
        data['trend_direction'] = np.where(data['sma20'] > data['sma50'], 'uptrend',
                                           np.where(data['sma20'] < data['sma50'], 'downtrend', 'sideways'))

        # Визначаємо поточний тренд (останній запис)
        current_trend = data['trend_direction'].iloc[-1]

        # Знаходимо початок поточного тренду
        trend_changes = data['trend_direction'] != data['trend_direction'].shift(1)
        trend_change_indices = trend_changes[trend_changes].index

        if len(trend_change_indices) > 0:
            # Знаходимо останню зміну тренду
            last_change_index = data.index.get_loc(trend_change_indices[-1])
            trend_start_index = last_change_index
        else:
            # Якщо змін тренду не було, початок - перший елемент датафрейму
            trend_start_index = 0

        # Рахуємо тривалість поточного тренду
        trend_periods = len(data) - trend_start_index

        # Створюємо допоміжний стовпець для групування періодів тренду
        data['trend_group'] = (data['trend_direction'] != data['trend_direction'].shift(1)).cumsum()

        # Групуємо за трендом і рахуємо тривалість кожного тренду
        trend_durations = data.groupby('trend_group').size()

        # Знаходимо максимальну тривалість тренду
        longest_streak = trend_durations.max() if not trend_durations.empty else 1

        # Рахуємо середню тривалість тренду
        avg_trend_duration = trend_durations.mean() if not trend_durations.empty else len(data)

        # Отримуємо дату початку поточного тренду
        if hasattr(data.index, 'strftime'):
            trend_start_date = data.index[trend_start_index].strftime('%Y-%m-%d')
        else:
            trend_start_date = str(trend_start_index)

        # Формуємо результат
        result = {
            'current_trend': current_trend,
            'current_trend_duration': trend_periods,
            'longest_trend_duration': int(longest_streak),
            'average_trend_duration': int(avg_trend_duration),
            'trend_start_date': trend_start_date,
            'total_periods_analyzed': len(data)
        }

        return result

    def identify_market_regime(self, data: pd.DataFrame) -> str:
        """
        Визначає поточний ринковий режим на основі технічних індикаторів.

        Метод аналізує комбінацію волатильності (ATR), сили та напрямку тренду (ADX, +DI, -DI),
        стану консолідації або розширення (Bollinger Bands), моментуму (MACD) та рівнів
        перекупленості / перепроданості (RSI), щоб класифікувати ринок у певну категорію.

        Для обчислення індикаторів використовуються бібліотеки `pandas_ta` або альтернативні обчислення вручну.

        Повертає один з наступних ринкових режимів:
            - 'strong_uptrend_high_volatility'
            - 'strong_downtrend_high_volatility'
            - 'uptrend_breakout'
            - 'downtrend_breakout'
            - 'consolidation'
            - 'range_bound'
            - 'low_volatility_sideways'
            - 'emerging_uptrend'
            - 'emerging_downtrend'
            - 'unknown'
            - 'analysis_error'

        У разі відсутності достатньої кількості даних або виникнення виключень, метод повертає 'analysis_error'.

        Parameters
        ----------
        data : pd.DataFrame
            Таблиця OHLCV-даних, яка повинна містити колонки 'high', 'low', 'close'.

        Returns
        -------
        str
            Опис поточного ринкового режиму.
        """
        try:
            # Перевірка наявності необхідних даних
            if 'close' not in data.columns:
                raise ValueError("DataFrame повинен містити стовпець 'close'")

            if 'high' not in data.columns or 'low' not in data.columns:
                raise ValueError("DataFrame повинен містити стовпці 'high' та 'low'")

            # Початкова перевірка кількості даних
            if len(data) < 50:
                self.logger.warning(f"Insufficient data for market regime analysis: {len(data)} rows")
                return "insufficient_data"

            # Копіюємо дані, щоб уникнути проблем з попередженнями pandas
            df = data.copy().sort_index()

            self.logger.debug(f"Starting market regime analysis with {len(df)} data points")

            # 1. Розрахунок індикаторів за допомогою pandas_ta з обробкою помилок

            try:
                # Розрахунок волатільності (ATR - Average True Range)
                atr_result = df.ta.atr(length=14)
                if atr_result is not None:
                    df['atr14'] = atr_result
                else:
                    # Fallback розрахунок ATR
                    df['tr'] = np.maximum(
                        df['high'] - df['low'],
                        np.maximum(
                            abs(df['high'] - df['close'].shift(1)),
                            abs(df['low'] - df['close'].shift(1))
                        )
                    )
                    df['atr14'] = df['tr'].rolling(window=14).mean()
            except Exception as e:
                self.logger.warning(f"ATR calculation failed: {e}, using fallback")
                df['tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                df['atr14'] = df['tr'].rolling(window=14).mean()

            # Нормалізована волатільність (ATR / Ціна)
            df['norm_volatility'] = (df['atr14'] / df['close']) * 100

            try:
                # Смуги Боллінджера
                bb = df.ta.bbands(length=20, std=2)
                if bb is not None and len(bb.columns) >= 3:
                    df['bb_upper'] = bb.iloc[:, 0]  # Верхня смуга
                    df['bb_middle'] = bb.iloc[:, 1]  # Середня лінія (SMA)
                    df['bb_lower'] = bb.iloc[:, 2]  # Нижня смуга
                else:
                    # Fallback розрахунок Bollinger Bands
                    sma20 = df['close'].rolling(window=20).mean()
                    std20 = df['close'].rolling(window=20).std()
                    df['bb_middle'] = sma20
                    df['bb_upper'] = sma20 + (std20 * 2)
                    df['bb_lower'] = sma20 - (std20 * 2)
            except Exception as e:
                self.logger.warning(f"Bollinger Bands calculation failed: {e}, using fallback")
                sma20 = df['close'].rolling(window=20).mean()
                std20 = df['close'].rolling(window=20).std()
                df['bb_middle'] = sma20
                df['bb_upper'] = sma20 + (std20 * 2)
                df['bb_lower'] = sma20 - (std20 * 2)

            # Ширина смуг Боллінджера відносно ціни
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100

            try:
                # ADX - Індекс спрямованого руху
                adx = df.ta.adx(length=14)
                if adx is not None and len(adx.columns) >= 3:
                    df['adx'] = adx[f'ADX_{14}']
                    df['+di'] = adx[f'DMP_{14}']
                    df['-di'] = adx[f'DMN_{14}']
                else:
                    # Fallback ADX calculation
                    df = self._calculate_adx_manual(df, 14)
            except Exception as e:
                self.logger.warning(f"ADX calculation failed: {e}, using manual calculation")
                df = self._calculate_adx_manual(df, 14)

            # 2. Додаткові індикатори для визначення ринкового стану

            try:
                # RSI для визначення перекупленості/перепроданості
                rsi_result = df.ta.rsi(length=14)
                if rsi_result is not None:
                    df['rsi'] = rsi_result
                else:
                    # Fallback RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
            except Exception as e:
                self.logger.warning(f"RSI calculation failed: {e}, using fallback")
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))

            try:
                # MACD для визначення моментуму
                macd = df.ta.macd(fast=12, slow=26, signal=9)
                if macd is not None and len(macd.columns) >= 3:
                    df['macd'] = macd[f'MACD_{12}_{26}_{9}']
                    df['macd_signal'] = macd[f'MACDs_{12}_{26}_{9}']
                    df['macd_hist'] = macd[f'MACDh_{12}_{26}_{9}']
                else:
                    # Fallback MACD
                    ema_fast = df['close'].ewm(span=12).mean()
                    ema_slow = df['close'].ewm(span=26).mean()
                    df['macd'] = ema_fast - ema_slow
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
            except Exception as e:
                self.logger.warning(f"MACD calculation failed: {e}, using fallback")
                ema_fast = df['close'].ewm(span=12).mean()
                ema_slow = df['close'].ewm(span=26).mean()
                df['macd'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']

            # 3. ВИПРАВЛЕНО: Розумна обробка NA значень

            # Замість видалення всіх рядків з NA, заповнюємо їх розумними значеннями
            numeric_columns = df.select_dtypes(include=[np.number]).columns

            # Заповнюємо NA значення методом forward fill, потім backward fill
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')

            # Для індикаторів, які потребують мінімальну кількість періодів,
            # беремо останні рядки з валідними даними
            required_indicators = ['adx', '+di', '-di', 'norm_volatility', 'bb_width', 'rsi', 'macd_hist']

            # Знаходимо останній рядок, де всі необхідні індикатори мають валідні значення
            valid_mask = df[required_indicators].notna().all(axis=1)
            valid_data = df[valid_mask]

            if len(valid_data) < 10:
                self.logger.warning(f"Only {len(valid_data)} rows with complete indicators")
                return "insufficient_data"

            # Використовуємо останні валідні дані для аналізу
            df_analysis = valid_data.tail(100) if len(valid_data) >= 100 else valid_data

            self.logger.debug(f"Using {len(df_analysis)} rows for market regime analysis")

            # 4. Аналіз поточного стану ринку

            # Визначаємо сучасний стан індикаторів
            adx_value = df_analysis['adx'].iloc[-1]
            plus_di = df_analysis['+di'].iloc[-1]
            minus_di = df_analysis['-di'].iloc[-1]

            # Середнє значення нормалізованої волатільності за останні 20 періодів
            recent_periods = min(20, len(df_analysis))
            recent_volatility = df_analysis['norm_volatility'].tail(recent_periods).mean()
            historical_volatility = df_analysis['norm_volatility'].mean()

            # Визначаємо квартилі волатільності з мінімальною кількістю даних
            if len(df_analysis) >= 20:
                low_vol_threshold = df_analysis['norm_volatility'].quantile(0.25)
                high_vol_threshold = df_analysis['norm_volatility'].quantile(0.75)
            else:
                # Використовуємо стандартні відхилення як альтернативу
                vol_mean = df_analysis['norm_volatility'].mean()
                vol_std = df_analysis['norm_volatility'].std()
                low_vol_threshold = vol_mean - vol_std
                high_vol_threshold = vol_mean + vol_std

            # Аналіз ширини смуг Боллінджера за останні 10 періодів
            bb_periods = min(10, len(df_analysis))
            recent_bb_width = df_analysis['bb_width'].tail(bb_periods).mean()
            historical_bb_width = df_analysis['bb_width'].mean()

            # Додаткові метрики
            rsi_value = df_analysis['rsi'].iloc[-1]
            macd_periods = min(3, len(df_analysis))
            macd_hist = df_analysis['macd_hist'].tail(macd_periods).mean()

            # 5. Визначення режиму ринку на основі комбінації показників

            # Визначаємо напрямок тренду
            if plus_di > minus_di:
                trend_direction = "uptrend"
            elif minus_di > plus_di:
                trend_direction = "downtrend"
            else:
                trend_direction = "neutral"

            # Визначаємо силу тренду на основі ADX
            if adx_value > 30:
                trend_strength = "strong"
            elif adx_value > 20:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"

            # Визначаємо стан волатільності
            if recent_volatility > high_vol_threshold * 1.5:
                volatility_state = "extremely_high"
            elif recent_volatility > high_vol_threshold:
                volatility_state = "high"
            elif recent_volatility < low_vol_threshold:
                volatility_state = "low"
            else:
                volatility_state = "normal"

            # Визначаємо стан ширини смуг Боллінджера (консолідація чи розширення)
            if historical_bb_width > 0:  # Перевіряємо, щоб уникнути ділення на нуль
                bb_ratio = recent_bb_width / historical_bb_width
                if bb_ratio < 0.7:
                    bollinger_state = "tight"  # Значна консолідація
                elif bb_ratio < 0.9:
                    bollinger_state = "narrowing"  # Звуження
                elif bb_ratio > 1.3:
                    bollinger_state = "wide"  # Значне розширення
                elif bb_ratio > 1.1:
                    bollinger_state = "expanding"  # Розширення
                else:
                    bollinger_state = "normal"  # Нормальний стан
            else:
                bollinger_state = "normal"

            # 6. Інтегрована оцінка режиму ринку з логуванням для діагностики

            self.logger.debug(
                f"Market regime indicators: ADX={adx_value:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f}, "
                f"RSI={rsi_value:.1f}, Volatility={recent_volatility:.2f}, "
                f"Trend: {trend_direction}({trend_strength}), BB: {bollinger_state}, Vol: {volatility_state}"
            )

            # Сильний тренд
            if trend_strength == "strong":
                if volatility_state in ["high", "extremely_high"]:
                    if trend_direction == "uptrend":
                        return "strong_uptrend_high_volatility"
                    else:
                        return "strong_downtrend_high_volatility"
                else:
                    if trend_direction == "uptrend":
                        return "strong_uptrend_normal_volatility"
                    else:
                        return "strong_downtrend_normal_volatility"

            # Помірний тренд
            elif trend_strength == "moderate":
                if bollinger_state in ["expanding", "wide"]:
                    if trend_direction == "uptrend":
                        return "emerging_uptrend"
                    else:
                        return "emerging_downtrend"
                else:
                    return "moderate_trend"

            # Слабкий тренд або відсутність тренду
            else:
                if bollinger_state in ["tight", "narrowing"]:
                    # Перевіряємо додаткові ознаки потенційного прориву
                    if rsi_value > 60 or rsi_value < 40:  # RSI наближається до екстремальних значень
                        return "accumulation_before_breakout"
                    else:
                        return "consolidation"

                elif volatility_state == "low":
                    return "low_volatility_sideways"

                elif bollinger_state in ["expanding", "wide"] and volatility_state in ["high", "extremely_high"]:
                    return "high_volatility_range"

                else:
                    return "choppy_market"

        except Exception as e:
            self.logger.error(f"Error in market regime identification: {str(e)}")
            return "analysis_error"