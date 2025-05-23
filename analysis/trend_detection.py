import json
from datetime import datetime
from scipy.stats import linregress
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy import stats
from data.db import DatabaseManager
import pandas_ta as ta
from utils.logger import CryptoLogger

class TrendDetection:


    def __init__(self, config=None):

        self.db_manager = DatabaseManager()

        # Ініціалізація конфігурації
        self.config = config or {}

        # Встановлення значень за замовчуванням, якщо не вказано інше
        self.default_window = self.config.get('default_window', 14)
        self.default_threshold = self.config.get('default_threshold', 0.02)
        self.min_points_for_level = self.config.get('min_points_for_level', 3)

        # Ініціалізація логера
        self.logger = CryptoLogger('INFO')

    def detect_trend(self, data: pd.DataFrame, window_size: int = 14) -> str:
        try:
            self.logger.debug(f"Input DataFrame size: {len(data)}")
            # Перевірка наявності даних
            if data.empty:
                self.logger.warning("Empty DataFrame provided for trend detection")
                return "unknown"

            # Перевірка, що data містить необхідний стовпець з ціною закриття
            if 'close' not in data.columns:
                self.logger.error("DataFrame must contain 'close' column for trend detection")
                raise ValueError("DataFrame must contain 'close' column")

            # Перевірка наявності необхідних стовпців
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"DataFrame must contain '{col}' column")
                    raise ValueError(f"DataFrame must contain '{col}' column")

            df = data.copy()

            # 1. Розрахунок ADX (Average Directional Index) для визначення сили тренду
            # Розрахунок True Range (TR)
            df['high_low'] = df['high'] - df['low']
            df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
            df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=window_size).mean()

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
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=window_size).mean() / df['atr'])
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=window_size).mean() / df['atr'])

            # Розрахунок DX і ADX
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            df['adx'] = df['dx'].rolling(window=window_size).mean()

            # 2. Розрахунок SMA (Simple Moving Average) для двох періодів
            df['sma_short'] = df['close'].rolling(window=window_size // 2).mean()
            df['sma_long'] = df['close'].rolling(window=window_size).mean()

            # 3. Розрахунок MACD (Moving Average Convergence Divergence)
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # 4. Розрахунок RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window_size).mean()
            avg_loss = loss.rolling(window=window_size).mean()

            # Уникаємо ділення на нуль
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))

            # Беремо останні дані для аналізу
            recent = df.tail(3)  # Використовуємо останні 3 свічки з повними індикаторами

            if len(recent) < 3:
                self.logger.warning("Not enough data points for reliable trend detection")
                return "unknown"

            # Визначення тренду на основі комбінації індикаторів
            trend_signals = []

            # Перевірка ADX (сила тренду)
            last_adx = recent['adx'].iloc[-1]
            last_plus_di = recent['plus_di'].iloc[-1]
            last_minus_di = recent['minus_di'].iloc[-1]

            adx_threshold = self.config.get('adx_threshold', 25)

            # ADX > 25 зазвичай вказує на наявність тренду
            if last_adx > adx_threshold:
                if last_plus_di > last_minus_di:
                    trend_signals.append("uptrend")
                else:
                    trend_signals.append("downtrend")
            else:
                trend_signals.append("sideways")

            # Перевірка SMA (ковзні середні)
            if recent['close'].iloc[-1] > recent['sma_short'].iloc[-1] > recent['sma_long'].iloc[-1]:
                trend_signals.append("uptrend")
            elif recent['close'].iloc[-1] < recent['sma_short'].iloc[-1] < recent['sma_long'].iloc[-1]:
                trend_signals.append("downtrend")
            else:
                trend_signals.append("sideways")

            # Перевірка MACD
            if recent['macd'].iloc[-1] > recent['macd_signal'].iloc[-1] and recent['macd_hist'].iloc[-1] > 0:
                trend_signals.append("uptrend")
            elif recent['macd'].iloc[-1] < recent['macd_signal'].iloc[-1] and recent['macd_hist'].iloc[-1] < 0:
                trend_signals.append("downtrend")
            else:
                trend_signals.append("sideways")

            # Перевірка RSI
            last_rsi = recent['rsi'].iloc[-1]
            if last_rsi > 60:  # Тенденція до перекупленості, але все ще в бичачому тренді
                trend_signals.append("uptrend")
            elif last_rsi < 40:  # Тенденція до перепроданості, але все ще в ведмежому тренді
                trend_signals.append("downtrend")
            else:  # RSI в нейтральній зоні
                trend_signals.append("sideways")

            # Логуємо сигнали для діагностики
            self.logger.debug(
                f"Trend signals: {trend_signals}, ADX: {last_adx}, +DI: {last_plus_di}, -DI: {last_minus_di}, RSI: {last_rsi}")

            # Визначаємо остаточний тренд на основі більшості сигналів
            uptrend_count = trend_signals.count("uptrend")
            downtrend_count = trend_signals.count("downtrend")
            sideways_count = trend_signals.count("sideways")

            if uptrend_count > downtrend_count and uptrend_count > sideways_count:
                return "uptrend"
            elif downtrend_count > uptrend_count and downtrend_count > sideways_count:
                return "downtrend"
            else:
                return "sideways"

        except Exception as e:
            self.logger.error(f"Error in trend detection: {str(e)}")
            return "unknown"

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:

        try:
            # Перевірка наявності необхідних стовпців
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"DataFrame must contain '{col}' column for ADX calculation")
                    raise ValueError(f"DataFrame must contain '{col}' column")

            # Копіюємо DataFrame щоб уникнути попереджень
            df = data.copy()
            # Розрахунок ADX з використанням pandas_ta
            adx_result = ta.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=period
            )

            # Додаємо результати до вхідного DataFrame
            df['adx'] = adx_result[f'ADX_{period}']
            df['plus_di'] = adx_result[f'DMP_{period}']  # +DI (Positive Directional Indicator)
            df['minus_di'] = adx_result[f'DMN_{period}']  # -DI (Negative Directional Indicator)

            # Логуємо успішне виконання
            self.logger.debug(f"ADX calculated successfully for {len(df)} data points with period {period}")

            return df

        except Exception as e:
            self.logger.error(f"Error in ADX calculation: {str(e)}")
            # Повертаємо вхідні дані без змін у випадку помилки
            return data

    def identify_support_resistance(self, data: pd.DataFrame,
                                    window_size: int = 20,
                                    threshold: float = 0.02) -> Dict[str, List[float]]:

        try:
            # Перевірка наявності даних
            if data.empty:
                self.logger.warning("Empty DataFrame provided for support/resistance detection")
                return {"support": [], "resistance": []}

            # Перевірка наявності цінових стовпців
            if 'high' not in data.columns or 'low' not in data.columns:
                # Якщо є тільки 'close', використовуємо його для обох
                if 'close' in data.columns:
                    self.logger.warning("Using 'close' prices for support/resistance as high/low not available")
                    high_prices = low_prices = data['close']
                else:
                    self.logger.error("DataFrame must contain 'high' and 'low' columns")
                    raise ValueError("Missing required price columns")
            else:
                high_prices = data['high']
                low_prices = data['low']

            # Знаходження локальних мінімумів (підтримка)
            local_mins = []
            for i in range(window_size, len(data) - window_size):
                if all(low_prices.iloc[i] <= low_prices.iloc[i - j] for j in range(1, window_size + 1)) and \
                        all(low_prices.iloc[i] <= low_prices.iloc[i + j] for j in range(1, window_size + 1)):
                    local_mins.append(low_prices.iloc[i])

            # Знаходження локальних максимумів (опір)
            local_maxs = []
            for i in range(window_size, len(data) - window_size):
                if all(high_prices.iloc[i] >= high_prices.iloc[i - j] for j in range(1, window_size + 1)) and \
                        all(high_prices.iloc[i] >= high_prices.iloc[i + j] for j in range(1, window_size + 1)):
                    local_maxs.append(high_prices.iloc[i])

            # Групування близьких рівнів підтримки
            grouped_support = self._group_price_levels(local_mins, threshold)

            # Групування близьких рівнів опору
            grouped_resistance = self._group_price_levels(local_maxs, threshold)

            # Формування результату
            result = {
                "support": grouped_support,
                "resistance": grouped_resistance
            }

            self.logger.info(
                f"Identified {len(grouped_support)} support and {len(grouped_resistance)} resistance levels")
            return result

        except Exception as e:
            self.logger.error(f"Error in support/resistance detection: {str(e)}")
            return {"support": [], "resistance": []}

    def _group_price_levels(self, price_points: List[float], threshold: float) -> List[float]:

        if not price_points:
            return []

        # Сортування точок
        sorted_points = sorted(price_points)

        # Групування близьких рівнів
        groups = []
        current_group = [sorted_points[0]]

        for i in range(1, len(sorted_points)):
            # Обчислення відносної різниці
            relative_diff = abs(sorted_points[i] - current_group[0]) / current_group[0]

            # Якщо точка близька до поточної групи, додаємо її до групи
            if relative_diff <= threshold:
                current_group.append(sorted_points[i])
            # Інакше завершуємо поточну групу і починаємо нову
            else:
                # Додаємо середнє значення групи до результату, якщо група має достатньо точок
                if len(current_group) >= self.min_points_for_level:
                    groups.append(sum(current_group) / len(current_group))
                # Починаємо нову групу
                current_group = [sorted_points[i]]

        # Додаємо останню групу, якщо вона має достатньо точок
        if len(current_group) >= self.min_points_for_level:
            groups.append(sum(current_group) / len(current_group))

        return groups

    def detect_breakouts(self, data: pd.DataFrame,
                         support_resistance: Dict[str, List[float]],
                         threshold: float = 0.01) -> List[Dict]:

        if data.empty or not support_resistance:
            return []

        # Переконаємося, що необхідні колонки присутні
        required_columns = ['close', 'high', 'low', 'date']
        if not all(col in data.columns for col in ['close', 'high', 'low']):
            raise ValueError("Дані повинні містити колонки 'close', 'high', 'low'")

        # Якщо немає колонки 'date', використаємо індекс
        if 'date' not in data.columns:
            data = data.reset_index()
            if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index

        breakouts = []

        # Отримуємо рівні підтримки та опору
        support_levels = sorted(support_resistance.get('support', []))
        resistance_levels = sorted(support_resistance.get('resistance', []))

        # Перевіряємо пробій кожного рівня опору
        for level in resistance_levels:
            # Знаходимо дні, коли ціна закриття була нижче рівня, а потім пробила вгору
            for i in range(1, len(data)):
                # Перевіряємо, чи ціна закриття була нижче рівня, а потім пробила його вище порогу
                if (data['close'].iloc[i - 1] < level and
                        data['close'].iloc[i] > level * (1 + threshold)):

                    breakout = {
                        'type': 'resistance_breakout',
                        'level': level,
                        'date': data['date'].iloc[i],
                        'price': data['close'].iloc[i],
                        'strength': (data['close'].iloc[i] - level) / level,  # Відносна сила пробою
                        'volume_change': None  # Можна додати динаміку об'єму, якщо доступно
                    }

                    # Додаємо зміну об'єму, якщо дані доступні
                    if 'volume' in data.columns:
                        volume_change = (data['volume'].iloc[i] / data['volume'].iloc[i - 1]) - 1
                        breakout['volume_change'] = volume_change

                    breakouts.append(breakout)

        # Перевіряємо пробій кожного рівня підтримки
        for level in support_levels:
            # Знаходимо дні, коли ціна закриття була вище рівня, а потім пробила вниз
            for i in range(1, len(data)):
                # Перевіряємо, чи ціна закриття була вище рівня, а потім пробила його нижче порогу
                if (data['close'].iloc[i - 1] > level and
                        data['close'].iloc[i] < level * (1 - threshold)):

                    breakout = {
                        'type': 'support_breakout',
                        'level': level,
                        'date': data['date'].iloc[i],
                        'price': data['close'].iloc[i],
                        'strength': (level - data['close'].iloc[i]) / level,  # Відносна сила пробою
                        'volume_change': None  # Можна додати динаміку об'єму, якщо доступно
                    }

                    # Додаємо зміну об'єму, якщо дані доступні
                    if 'volume' in data.columns:
                        volume_change = (data['volume'].iloc[i] / data['volume'].iloc[i - 1]) - 1
                        breakout['volume_change'] = volume_change

                    breakouts.append(breakout)

        # Сортуємо пробої за датою
        breakouts = sorted(breakouts, key=lambda x: x['date'])

        return breakouts
    def find_swing_points(self, data: pd.DataFrame, window_size: int = 5) -> Dict[str, List[Dict]]:

        if data.empty or len(data) < window_size * 2:
            return {'highs': [], 'lows': []}

        # Перевіряємо наявність необхідних колонок
        required_columns = ['high', 'low', 'date']
        if not all(col in data.columns for col in ['high', 'low']):
            raise ValueError("Дані повинні містити колонки 'high' та 'low'")

        # Якщо немає колонки 'date', використаємо індекс
        if 'date' not in data.columns:
            data = data.reset_index()
            if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index

        swing_points = {'highs': [], 'lows': []}

        # Для кожної точки перевіряємо, чи є вона локальним екстремумом
        for i in range(window_size, len(data) - window_size):
            # Перевіряємо, чи є точка локальним максимумом
            is_high = True
            for j in range(1, window_size + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i - j] or \
                        data['high'].iloc[i] <= data['high'].iloc[i + j]:
                    is_high = False
                    break

            if is_high:
                # Якщо точка є локальним максимумом
                swing_high = {
                    'date': data['date'].iloc[i],
                    'price': data['high'].iloc[i],
                    'index': i,
                    'strength': 0  # Ініціалізуємо силу точки
                }

                # Розраховуємо силу точки (наскільки вона виділяється)
                left_diff = min(data['high'].iloc[i] - data['high'].iloc[i - j] for j in range(1, window_size + 1))
                right_diff = min(data['high'].iloc[i] - data['high'].iloc[i + j] for j in range(1, window_size + 1))
                swing_high['strength'] = min(left_diff, right_diff) / data['high'].iloc[i]

                # Додаємо додаткові дані, якщо доступні
                if 'volume' in data.columns:
                    swing_high['volume'] = data['volume'].iloc[i]

                swing_points['highs'].append(swing_high)

            # Перевіряємо, чи є точка локальним мінімумом
            is_low = True
            for j in range(1, window_size + 1):
                if data['low'].iloc[i] >= data['low'].iloc[i - j] or \
                        data['low'].iloc[i] >= data['low'].iloc[i + j]:
                    is_low = False
                    break

            if is_low:
                # Якщо точка є локальним мінімумом
                swing_low = {
                    'date': data['date'].iloc[i],
                    'price': data['low'].iloc[i],
                    'index': i,
                    'strength': 0  # Ініціалізуємо силу точки
                }

                # Розраховуємо силу точки (наскільки вона виділяється)
                left_diff = min(data['low'].iloc[i - j] - data['low'].iloc[i] for j in range(1, window_size + 1))
                right_diff = min(data['low'].iloc[i + j] - data['low'].iloc[i] for j in range(1, window_size + 1))
                swing_low['strength'] = min(left_diff, right_diff) / data['low'].iloc[i]

                # Додаємо додаткові дані, якщо доступні
                if 'volume' in data.columns:
                    swing_low['volume'] = data['volume'].iloc[i]

                swing_points['lows'].append(swing_low)

        return swing_points

    def calculate_trend_strength(self, data: pd.DataFrame) -> float:

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

            # 1. Розрахунок ADX (Average Directional Index) - основний індикатор сили тренду
            adx_value = 0.5  # Нейтральне значення за замовчуванням

            if all(col in df.columns for col in ['high', 'low', 'close']):
                # Розрахунок ADX власноруч
                high = df['high']
                low = df['low']
                close = df['close']
                length = 14  # Стандартний період для ADX

                # True Range (TR)
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr = true_range.rolling(window=length).mean()

                # +DM і -DM (Directional Movement)
                plus_dm = high - high.shift(1)
                minus_dm = low.shift(1) - low

                plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
                minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

                # Smoothed +DM і -DM
                plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr)

                # Розрахунок ADX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=length).mean()

                adx_value = min(1.0, adx.iloc[-1] / 100.0) if not np.isnan(adx.iloc[-1]) else 0.5
            else:
                self.logger.warning("No high/low data for ADX calculation, using default value")

            # 2. Розрахунок лінійної регресії для визначення напрямку і сили тренду
            x = np.arange(len(df))
            y = df['close'].values
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            r_squared = r_value ** 2  # Коефіцієнт детермінації (0-1)
            trend_direction = np.sign(slope)  # 1 для висхідного, -1 для низхідного

            # 3. Розрахунок Aroon-індикатору (показує силу і напрямок тренду)
            aroon_strength = 0.5  # Значення за замовчуванням

            if all(col in df.columns for col in ['high', 'low']):
                length = 14  # Стандартний період для Aroon

                # Aroon Up: 100 * (length - periods since highest high) / length
                highest_idx = df['high'].rolling(window=length).apply(lambda x: x.argmax())
                periods_since_high = np.arange(len(df)) - highest_idx
                aroon_up = 100 * (length - periods_since_high) / length

                # Aroon Down: 100 * (length - periods since lowest low) / length
                lowest_idx = df['low'].rolling(window=length).apply(lambda x: x.argmin())
                periods_since_low = np.arange(len(df)) - lowest_idx
                aroon_down = 100 * (length - periods_since_low) / length

                # Замінюємо NaN значення
                aroon_up = aroon_up.fillna(0)
                aroon_down = aroon_down.fillna(0)

                # Aroon Oscillator = Aroon Up - Aroon Down
                aroon_oscillator = (aroon_up.iloc[-1] - aroon_down.iloc[-1]) / 100.0  # від -1 до 1
                aroon_strength = abs(aroon_oscillator)  # сила тренду по Aroon (0-1)

            # 4. Розрахунок MACD для визначення імпульсу тренду
            fast_period = 12
            slow_period = 26
            signal_period = 9

            # Розрахунок EMA
            ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()

            # MACD Line = Fast EMA - Slow EMA
            macd_line = ema_fast - ema_slow

            # Signal Line = EMA of MACD Line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # MACD Histogram = MACD Line - Signal Line
            macd_hist = macd_line - signal_line

            # Нормалізований MACD (відносно ціни)
            avg_price = df['close'].mean()
            macd_norm = abs(macd_line.iloc[-1]) / avg_price
            macd_strength = min(1.0, macd_norm * 100)  # Обмежуємо до 1.0

            # 5. Розрахунок RSI для визначення сили тренду через перекупленість/перепроданість
            period = 14
            delta = df['close'].diff()

            # Отримуємо позитивні та негативні зміни
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Розраховуємо середні значення gain і loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # Розраховуємо RS і RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            rsi_value = rsi.iloc[-1]

            # Перетворюємо RSI на індикатор сили тренду (екстремальні значення = сильний тренд)
            rsi_strength = 0.0
            if rsi_value >= 70 or rsi_value <= 30:
                # Екстремальні значення RSI вказують на сильний тренд
                rsi_strength = (abs(rsi_value - 50) - 20) / 30  # Перетворюємо на діапазон 0-1
                rsi_strength = max(0, min(1, rsi_strength))  # Обмежуємо діапазоном 0-1
            else:
                # Значення RSI ближче до 50 вказують на слабший тренд
                rsi_strength = 0.2  # Базове значення для середнього діапазону

            # 6. Аналіз патернів свічок
            candle_pattern_strength = 0.0
            if all(col in df.columns for col in ['open', 'high', 'low']):
                # Визначаємо останні 3 свічки
                recent = df.tail(3)
                # Рахуємо тіла свічок (абсолютна різниця між відкриттям і закриттям)
                bodies = abs(recent['close'] - recent['open'])
                avg_body = bodies.mean()

                # Перевіряємо, чи мають свічки послідовний напрямок і великі тіла
                consistent_direction = all(np.sign(recent['close'] - recent['open']) == trend_direction)
                large_bodies = all(bodies > avg_body * 0.8)

                if consistent_direction and large_bodies:
                    candle_pattern_strength = 0.2

            # 7. Розрахунок волатильності (стандартне відхилення відносних змін)
            volatility = df['close'].pct_change().std()
            volatility_factor = 1.0 - min(1.0, volatility * 20)  # Висока волатильність зменшує силу тренду

            # 8. Аналіз об'єму (якщо доступний)
            volume_factor = 0.5  # Нейтральне значення за замовчуванням
            if 'volume' in df.columns:
                # Розрахунок SMA для об'єму
                volume_sma = df['volume'].rolling(window=20).mean()
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
                'volatility': 0.10,  # Фактор волатильності (зворотний)
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

    def detect_trend_reversal(self, data: pd.DataFrame) -> List[Dict]:

        if data.empty or len(data) < 30:
            return []

        # Перевіряємо наявність необхідних колонок
        required_columns = ['close', 'high', 'low', 'open']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {', '.join(required_columns)}")

        # Підготовка даних
        if 'date' not in data.columns:
            data = data.reset_index()
            if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index

        # Копіюємо дані, щоб уникнути проблем з попередженнями pandas
        df = data.copy()

        # 1. Обчислюємо ковзні середні для визначення тренду
        df['sma20'] = ta.sma(df['close'], length=20)
        df['sma50'] = ta.sma(df['close'], length=50)

        # 2. Обчислюємо RSI для визначення перекупленості/перепроданості
        rsi = ta.rsi(df['close'], length=14)
        df['rsi'] = rsi

        # 3. Обчислюємо MACD для підтвердження розвороту
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']

        # 4. Обчислюємо Bollinger Bands для визначення рівнів підтримки/опору
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']

        # 5. Додаємо основні свічкові патерни
        def identify_engulfing(df, i):
            # Бичаче поглинання
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # Поточна свічка бича
                    df['close'].iloc[i - 1] < df['open'].iloc[i - 1] and  # Попередня свічка ведмежа
                    df['close'].iloc[i] > df['open'].iloc[i - 1] and  # Поточне закриття вище попереднього відкриття
                    df['open'].iloc[i] < df['close'].iloc[i - 1]):  # Поточне відкриття нижче попереднього закриття
                return 'bullish'
            # Ведмеже поглинання
            elif (df['close'].iloc[i] < df['open'].iloc[i] and  # Поточна свічка ведмежа
                  df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and  # Попередня свічка бича
                  df['close'].iloc[i] < df['open'].iloc[i - 1] and  # Поточне закриття нижче попереднього відкриття
                  df['open'].iloc[i] > df['close'].iloc[i - 1]):  # Поточне відкриття вище попереднього закриття
                return 'bearish'
            return None

        def identify_hammer(df, i):
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])

            # Якщо нижня тінь хоча б в 2 рази більша за тіло та верхня тінь маленька
            if lower_shadow > 2 * body_size and upper_shadow < 0.5 * body_size:
                if df['close'].iloc[i] > df['open'].iloc[i]:  # Бича свічка
                    return True
            return False

        def identify_shooting_star(df, i):
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])

            # Якщо верхня тінь хоча б в 2 рази більша за тіло та нижня тінь маленька
            if upper_shadow > 2 * body_size and lower_shadow < 0.5 * body_size:
                if df['close'].iloc[i] < df['open'].iloc[i]:  # Ведмежа свічка
                    return True
            return False


        # Ініціалізуємо список для зберігання результатів
        reversals = []

        # Аналізуємо можливі сигнали розвороту
        for i in range(50, len(df)):
            current_date = df['date'].iloc[i]
            current_price = df['close'].iloc[i]

            # Визначаємо поточний тренд за перетином ковзних середніх
            current_trend = 'uptrend' if df['sma20'].iloc[i] > df['sma50'].iloc[i] else 'downtrend'
            prev_trend = 'uptrend' if df['sma20'].iloc[i - 1] > df['sma50'].iloc[i - 1] else 'downtrend'

            reversal_signal = None
            signal_strength = 0.0
            signal_reasons = []

            # 1. Перетин ковзних середніх (сильний сигнал розвороту)
            if current_trend != prev_trend:
                if current_trend == 'downtrend':
                    reversal_signal = 'bearish_crossover'
                    signal_strength = 0.7
                    signal_reasons.append("SMA20 перетнула SMA50 знизу вгору")
                else:
                    reversal_signal = 'bullish_crossover'
                    signal_strength = 0.7
                    signal_reasons.append("SMA20 перетнула SMA50 зверху вниз")

            # 2. Перетин MACD з сигнальною лінією
            macd_cross_bullish = df['macd'].iloc[i] > df['macd_signal'].iloc[i] and df['macd'].iloc[i - 1] <= \
                                 df['macd_signal'].iloc[i - 1]
            macd_cross_bearish = df['macd'].iloc[i] < df['macd_signal'].iloc[i] and df['macd'].iloc[i - 1] >= \
                                 df['macd_signal'].iloc[i - 1]

            if macd_cross_bearish and current_trend == 'uptrend':
                if not reversal_signal:
                    reversal_signal = 'bearish_macd_cross'
                    signal_strength = 0.6
                    signal_reasons.append("Ведмежий перетин MACD")
                else:
                    signal_strength += 0.2
                    signal_reasons.append("Підтверджено ведмежим перетином MACD")

            elif macd_cross_bullish and current_trend == 'downtrend':
                if not reversal_signal:
                    reversal_signal = 'bullish_macd_cross'
                    signal_strength = 0.6
                    signal_reasons.append("Бичачий перетин MACD")
                else:
                    signal_strength += 0.2
                    signal_reasons.append("Підтверджено бичачим перетином MACD")

            # 3. RSI перекупленість/перепроданість
            if df['rsi'].iloc[i] > 70 and current_trend == 'uptrend':
                if df['rsi'].iloc[i] < df['rsi'].iloc[i - 1]:
                    if not reversal_signal:
                        reversal_signal = 'overbought'
                        signal_strength = 0.5
                        signal_reasons.append(f"RSI перекуплений ({df['rsi'].iloc[i]:.1f}) і почав падати")
                    else:
                        signal_strength += 0.15
                        signal_reasons.append(f"Підтверджено перекупленим RSI ({df['rsi'].iloc[i]:.1f})")

            elif df['rsi'].iloc[i] < 30 and current_trend == 'downtrend':
                if df['rsi'].iloc[i] > df['rsi'].iloc[i - 1]:
                    if not reversal_signal:
                        reversal_signal = 'oversold'
                        signal_strength = 0.5
                        signal_reasons.append(f"RSI перепроданий ({df['rsi'].iloc[i]:.1f}) і почав рости")
                    else:
                        signal_strength += 0.15
                        signal_reasons.append(f"Підтверджено перепроданим RSI ({df['rsi'].iloc[i]:.1f})")

            # 4. Свічкові патерни
            # Перевіряємо патерн поглинання
            engulfing = identify_engulfing(df, i)
            if engulfing == 'bearish' and current_trend == 'uptrend':
                if not reversal_signal:
                    reversal_signal = 'bearish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Ведмежий свічковий патерн: bearish_engulfing")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено ведмежим свічковим патерном: bearish_engulfing")
            elif engulfing == 'bullish' and current_trend == 'downtrend':
                if not reversal_signal:
                    reversal_signal = 'bullish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Бичачий свічковий патерн: bullish_engulfing")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено бичачим свічковим патерном: bullish_engulfing")

            # Перевіряємо патерн молот
            if identify_hammer(df, i) and current_trend == 'downtrend':
                if not reversal_signal:
                    reversal_signal = 'bullish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Бичачий свічковий патерн: hammer")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено бичачим свічковим патерном: hammer")

            # Перевіряємо патерн падаюча зірка
            if identify_shooting_star(df, i) and current_trend == 'uptrend':
                if not reversal_signal:
                    reversal_signal = 'bearish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Ведмежий свічковий патерн: shooting_star")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено ведмежим свічковим патерном: shooting_star")

            # 5. Тестування рівнів Bollinger Bands
            # Ціна пробиває верхню лінію і повертається назад (ведмежий сигнал)
            if current_trend == 'uptrend' and df['close'].iloc[i - 1] > df['bb_upper'].iloc[i - 1] and df['close'].iloc[
                i] < df['bb_upper'].iloc[i]:
                if not reversal_signal:
                    reversal_signal = 'bearish_bb_rejection'
                    signal_strength = 0.45
                    signal_reasons.append("Відбиття від верхньої лінії Bollinger Bands")
                else:
                    signal_strength += 0.1
                    signal_reasons.append("Підтверджено відбиттям від верхньої лінії Bollinger Bands")

            # Ціна пробиває нижню лінію і повертається назад (бичачий сигнал)
            elif current_trend == 'downtrend' and df['close'].iloc[i - 1] < df['bb_lower'].iloc[i - 1] and \
                    df['close'].iloc[i] > df['bb_lower'].iloc[i]:
                if not reversal_signal:
                    reversal_signal = 'bullish_bb_rejection'
                    signal_strength = 0.45
                    signal_reasons.append("Відбиття від нижньої лінії Bollinger Bands")
                else:
                    signal_strength += 0.1
                    signal_reasons.append("Підтверджено відбиттям від нижньої лінії Bollinger Bands")

            # Якщо виявлено сигнал розвороту, додаємо його до списку
            if reversal_signal:
                reversal = {
                    'date': current_date,
                    'price': current_price,
                    'signal_type': reversal_signal,
                    'strength': min(1.0, signal_strength),  # Обмежуємо силу сигналу до 1.0
                    'current_trend': current_trend,
                    'rsi': df['rsi'].iloc[i],
                    'reasons': signal_reasons
                }

                # Додаємо додаткові дані, якщо доступні
                if 'volume' in df.columns:
                    reversal['volume'] = df['volume'].iloc[i]

                reversals.append(reversal)

        return reversals

    def calculate_fibonacci_levels(self, data: pd.DataFrame, trend_type: str) -> Dict[str, float]:

        if data.empty or len(data) < 2:
            return {}

        # Перевіряємо наявність необхідних колонок
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {', '.join(required_columns)}")

        # Перевіряємо тип тренду
        if trend_type not in ['uptrend', 'downtrend']:
            raise ValueError("trend_type повинен бути 'uptrend' або 'downtrend'")

        # Використаємо вікно останніх 90 точок або всі дані, якщо їх менше
        window = min(90, len(data))
        recent_data = data.iloc[-window:]

        # Знаходимо значення екстремумів для розрахунку
        if trend_type == 'uptrend':
            # Для висхідного тренду беремо мінімум як стартову точку і максимум як кінцеву
            start_price = recent_data['low'].min()
            end_price = recent_data['high'].max()
        else:  # downtrend
            # Для низхідного тренду беремо максимум як стартову точку і мінімум як кінцеву
            start_price = recent_data['high'].max()
            end_price = recent_data['low'].min()

        # Використовуємо власну функцію для розрахунку рівнів Фібоначчі замість panads_ta
        # оскільки pandas_ta може не мати прямого еквіваленту для fibonacci
        retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        extension_levels = [1.272, 1.618, 2.618]

        # Розрахунок цінових рівнів на основі відрізка [start_price, end_price]
        price_range = end_price - start_price

        result = {}
        # Додаємо рівні ретрейсменту
        for level in retracement_levels:
            level_str = f"{level:.3f}" if level != 0 and level != 1 else f"{level:.1f}"
            if trend_type == 'uptrend':
                result[level_str] = end_price - (price_range * level)
            else:
                result[level_str] = start_price - (price_range * level)

        # Додаємо рівні екстенжн
        for level in extension_levels:
            level_str = f"{level:.3f}"
            if trend_type == 'uptrend':
                result[level_str] = end_price + (price_range * (level - 1))
            else:
                result[level_str] = start_price - (price_range * level)

        # Додаємо інформацію про використані екстремуми
        result['swing_high'] = recent_data['high'].max()
        result['swing_low'] = recent_data['low'].min()

        return result

    def detect_chart_patterns(self, data: pd.DataFrame) -> List[Dict]:

        if data.empty or len(data) < 30:  # Потрібно достатньо точок для аналізу
            return []

        # Перевіряємо наявність необхідних колонок
        required_columns = ['high', 'low', 'close', 'date']
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Дані повинні містити колонки 'high', 'low', 'close'")

        # Якщо немає колонки 'date', використаємо індекс
        if 'date' not in data.columns:
            data = data.reset_index()
            if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index

        # Знаходимо точки розвороту для аналізу паттернів
        swing_points = self.find_swing_points(data, window_size=3)

        # Виявлені паттерни будемо зберігати тут
        patterns = []

        # Отримуємо відсортовані за індексом точки розвороту
        highs = sorted(swing_points['highs'], key=lambda x: x['index'])
        lows = sorted(swing_points['lows'], key=lambda x: x['index'])

        # 1. Виявлення паттерну "Подвійне дно" (Double Bottom)
        if len(lows) >= 2:
            for i in range(len(lows) - 1):
                # Беремо два послідовних мінімума
                low1 = lows[i]
                low2 = lows[i + 1]

                # Перевіряємо, чи знаходяться вони приблизно на одному рівні
                price_diff = abs(low1['price'] - low2['price'])
                avg_price = (low1['price'] + low2['price']) / 2
                price_diff_pct = price_diff / avg_price

                # Критерії для подвійного дна:
                # - Два мінімуми приблизно на одному рівні (різниця менше 3%)
                # - Між ними повинен бути максимум, що принаймні на 3% вище
                # - Другий мінімум не повинен бути набагато нижче першого
                if price_diff_pct < 0.03:
                    # Шукаємо максимум між цими двома мінімумами
                    between_highs = [h for h in highs if low1['index'] < h['index'] < low2['index']]

                    if between_highs:
                        middle_high = max(between_highs, key=lambda x: x['price'])
                        # Перевіряємо, чи достатньо великий підйом між мінімумами
                        rise = (middle_high['price'] - min(low1['price'], low2['price'])) / min(low1['price'],
                                                                                                low2['price'])

                        if rise > 0.03:
                            # Знаходимо підтвердження паттерну - зростання після другого мінімуму
                            confirmation_point = None
                            potential_highs = [h for h in highs if h['index'] > low2['index']]

                            if potential_highs:
                                confirmation_point = potential_highs[0]

                                # Додаємо патерн, якщо є підтвердження
                                if confirmation_point and confirmation_point['price'] > middle_high['price']:
                                    pattern = {
                                        'type': 'double_bottom',
                                        'start_date': low1['date'],
                                        'end_date': confirmation_point['date'] if confirmation_point else low2['date'],
                                        'bottom1': {'date': low1['date'], 'price': low1['price']},
                                        'bottom2': {'date': low2['date'], 'price': low2['price']},
                                        'middle_peak': {'date': middle_high['date'], 'price': middle_high['price']},
                                        'confirmation': confirmation_point,
                                        'strength': min(1.0, rise * 10)  # Оцінка сили паттерна
                                    }
                                    patterns.append(pattern)

        # 2. Виявлення паттерну "Подвійна вершина" (Double Top)
        if len(highs) >= 2:
            for i in range(len(highs) - 1):
                # Беремо два послідовних максимума
                high1 = highs[i]
                high2 = highs[i + 1]

                # Перевіряємо, чи знаходяться вони приблизно на одному рівні
                price_diff = abs(high1['price'] - high2['price'])
                avg_price = (high1['price'] + high2['price']) / 2
                price_diff_pct = price_diff / avg_price

                # Критерії для подвійної вершини:
                # - Два максимуми приблизно на одному рівні (різниця менше 3%)
                # - Між ними повинен бути мінімум, що принаймні на 3% нижче
                # - Другий максимум не повинен бути набагато вище першого
                if price_diff_pct < 0.03:
                    # Шукаємо мінімум між цими двома максимумами
                    between_lows = [l for l in lows if high1['index'] < l['index'] < high2['index']]

                    if between_lows:
                        middle_low = min(between_lows, key=lambda x: x['price'])
                        # Перевіряємо, чи достатньо велике падіння між максимумами
                        drop = (min(high1['price'], high2['price']) - middle_low['price']) / middle_low['price']

                        if drop > 0.03:
                            # Знаходимо підтвердження паттерну - падіння після другого максимуму
                            confirmation_point = None
                            potential_lows = [l for l in lows if l['index'] > high2['index']]

                            if potential_lows:
                                confirmation_point = potential_lows[0]

                                # Додаємо патерн, якщо є підтвердження
                                if confirmation_point and confirmation_point['price'] < middle_low['price']:
                                    pattern = {
                                        'type': 'double_top',
                                        'start_date': high1['date'],
                                        'end_date': confirmation_point['date'] if confirmation_point else high2['date'],
                                        'top1': {'date': high1['date'], 'price': high1['price']},
                                        'top2': {'date': high2['date'], 'price': high2['price']},
                                        'middle_trough': {'date': middle_low['date'], 'price': middle_low['price']},
                                        'confirmation': confirmation_point,
                                        'strength': min(1.0, drop * 10)  # Оцінка сили паттерна
                                    }
                                    patterns.append(pattern)

        # 3. Виявлення паттерну "Голова і плечі" (Head and Shoulders)
        if len(highs) >= 3 and len(lows) >= 4:
            for i in range(len(highs) - 2):
                # Беремо три послідовних максимума для можливого паттерну голова-плечі
                left_shoulder = highs[i]
                head = highs[i + 1]
                right_shoulder = highs[i + 2]

                # Шукаємо відповідні "шийні" рівні (neckline)
                left_neck_candidates = [l for l in lows if
                                        l['index'] > left_shoulder['index'] and l['index'] < head['index']]
                right_neck_candidates = [l for l in lows if
                                         l['index'] > head['index'] and l['index'] < right_shoulder['index']]

                if left_neck_candidates and right_neck_candidates:
                    left_neck = min(left_neck_candidates, key=lambda x: abs(x['index'] - head['index']))
                    right_neck = min(right_neck_candidates, key=lambda x: abs(x['index'] - head['index']))

                    # Критерії для паттерну голова-плечі:
                    # - Середній максимум (голова) повинен бути вище крайніх (плечей)
                    # - Плечі повинні бути приблизно на одному рівні (різниця менше 5%)
                    # - Шийні рівні також повинні бути приблизно на одному рівні
                    if (head['price'] > left_shoulder['price'] and
                            head['price'] > right_shoulder['price']):

                        # Перевіряємо симетричність плечей
                        shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
                        avg_shoulder = (left_shoulder['price'] + right_shoulder['price']) / 2
                        shoulder_diff_pct = shoulder_diff / avg_shoulder

                        # Перевіряємо рівень "шиї"
                        neck_diff = abs(left_neck['price'] - right_neck['price'])
                        avg_neck = (left_neck['price'] + right_neck['price']) / 2
                        neck_diff_pct = neck_diff / avg_neck

                        if shoulder_diff_pct < 0.05 and neck_diff_pct < 0.05:
                            # Шукаємо підтвердження - пробій рівня шиї після правого плеча
                            neckline_level = (left_neck['price'] + right_neck['price']) / 2
                            confirmation_candidates = [l for l in lows if l['index'] > right_shoulder['index']]

                            if confirmation_candidates:
                                confirmation = confirmation_candidates[0]

                                # Перевіряємо, чи ціна пробила лінію шиї вниз
                                if confirmation['price'] < neckline_level:
                                    # Оцінка потенціалу падіння (відстань від голови до шиї)
                                    potential_drop = head['price'] - neckline_level
                                    # Оцінка поточного падіння
                                    current_drop = neckline_level - confirmation['price']
                                    # Сила паттерну - відношення поточного падіння до потенційного
                                    pattern_strength = min(1.0,
                                                           current_drop / potential_drop) if potential_drop > 0 else 0.5

                                    pattern = {
                                        'type': 'head_and_shoulders',
                                        'start_date': left_shoulder['date'],
                                        'end_date': confirmation['date'],
                                        'left_shoulder': {'date': left_shoulder['date'],
                                                          'price': left_shoulder['price']},
                                        'head': {'date': head['date'], 'price': head['price']},
                                        'right_shoulder': {'date': right_shoulder['date'],
                                                           'price': right_shoulder['price']},
                                        'neckline': neckline_level,
                                        'confirmation': {'date': confirmation['date'], 'price': confirmation['price']},
                                        'strength': pattern_strength
                                    }
                                    patterns.append(pattern)

        # 4. Виявлення паттерну "Трикутник" (Triangle)
        # Для цього потрібно проаналізувати послідовні максимуми і мінімуми
        # та перевірити, чи утворюють вони сходинки, що звужуються
        if len(highs) >= 3 and len(lows) >= 3:
            # Беремо останні точки для аналізу тренду останнього періоду
            last_highs = highs[-3:]
            last_lows = lows[-3:]

            # Перевіряємо на висхідний трикутник (ascending triangle):
            # - Горизонтальний опір (плоска верхня лінія)
            # - Висхідна лінія підтримки
            if (max(h['price'] for h in last_highs) - min(h['price'] for h in last_highs)) / max(
                    h['price'] for h in last_highs) < 0.03:
                # Перевіряємо, чи мінімуми зростають
                if all(last_lows[i]['price'] < last_lows[i + 1]['price'] for i in range(len(last_lows) - 1)):
                    # Знаходимо початок і кінець паттерну
                    start_date = min(last_highs[0]['date'], last_lows[0]['date'])
                    end_date = max(last_highs[-1]['date'], last_lows[-1]['date'])

                    # Визначаємо рівень опору
                    resistance_level = sum(h['price'] for h in last_highs) / len(last_highs)

                    pattern = {
                        'type': 'ascending_triangle',
                        'start_date': start_date,
                        'end_date': end_date,
                        'resistance_level': resistance_level,
                        'last_point': {'date': end_date, 'price': data.loc[data['date'] == end_date, 'close'].iloc[0]},
                        'strength': 0.7  # Фіксована оцінка сили для простоти
                    }
                    patterns.append(pattern)

            # Перевіряємо на низхідний трикутник (descending triangle):
            # - Горизонтальна підтримка (плоска нижня лінія)
            # - Низхідна лінія опору
            if (max(l['price'] for l in last_lows) - min(l['price'] for l in last_lows)) / max(
                    l['price'] for l in last_lows) < 0.03:
                # Перевіряємо, чи максимуми спадають
                if all(last_highs[i]['price'] > last_highs[i + 1]['price'] for i in range(len(last_highs) - 1)):
                    # Знаходимо початок і кінець паттерну
                    start_date = min(last_highs[0]['date'], last_lows[0]['date'])
                    end_date = max(last_highs[-1]['date'], last_lows[-1]['date'])

                    # Визначаємо рівень підтримки
                    support_level = sum(l['price'] for l in last_lows) / len(last_lows)

                    pattern = {
                        'type': 'descending_triangle',
                        'start_date': start_date,
                        'end_date': end_date,
                        'support_level': support_level,
                        'last_point': {'date': end_date, 'price': data.loc[data['date'] == end_date, 'close'].iloc[0]},
                        'strength': 0.7  # Фіксована оцінка сили для простоти
                    }
                    patterns.append(pattern)

            # Перевіряємо на симетричний трикутник (symmetric triangle):
            # - Низхідна лінія опору
            # - Висхідна лінія підтримки
            if (all(last_highs[i]['price'] > last_highs[i + 1]['price'] for i in range(len(last_highs) - 1)) and
                    all(last_lows[i]['price'] < last_lows[i + 1]['price'] for i in range(len(last_lows) - 1))):
                # Знаходимо початок і кінець паттерну
                start_date = min(last_highs[0]['date'], last_lows[0]['date'])
                end_date = max(last_highs[-1]['date'], last_lows[-1]['date'])

                pattern = {
                    'type': 'symmetric_triangle',
                    'start_date': start_date,
                    'end_date': end_date,
                    'upper_level': last_highs[-1]['price'],
                    'lower_level': last_lows[-1]['price'],
                    'last_point': {'date': end_date, 'price': data.loc[data['date'] == end_date, 'close'].iloc[0]},
                    'strength': 0.6  # Фіксована оцінка сили для простоти
                }
                patterns.append(pattern)


        return patterns

    def estimate_trend_duration(self, data: pd.DataFrame) -> Dict[str, int]:

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

        # Перевірка наявності необхідних даних
        if 'close' not in data.columns:
            raise ValueError("DataFrame повинен містити стовпець 'close'")

        if 'high' not in data.columns or 'low' not in data.columns:
            raise ValueError("DataFrame повинен містити стовпці 'high' та 'low'")

        # Копіюємо дані, щоб уникнути проблем з попередженнями pandas
        df = data.copy()

        # 1. Розрахунок індикаторів за допомогою pandas_ta

        # Розрахунок волатильності (ATR - Average True Range)
        df['atr14'] = df.ta.atr(length=14)

        # Нормалізована волатильність (ATR / Ціна)
        df['norm_volatility'] = df['atr14'] / df['close'] * 100

        # Смуги Боллінджера
        bb = df.ta.bbands(length=20, std=2)
        df['bb_upper'] = bb[f'BBU_{20}_2.0']
        df['bb_lower'] = bb[f'BBL_{20}_2.0']
        df['bb_middle'] = bb[f'BBM_{20}_2.0']

        # Ширина смуг Боллінджера відносно ціни
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100

        # ADX - Індекс спрямованого руху
        adx = df.ta.adx(length=14)
        df['adx'] = adx[f'ADX_{14}']
        df['+di'] = adx[f'DMP_{14}']
        df['-di'] = adx[f'DMN_{14}']

        # 2. Додаткові індикатори для визначення ринкового стану

        # Додаємо RSI для визначення перекупленості/перепроданості
        df['rsi'] = df.ta.rsi(length=14)

        # Додаємо MACD для визначення моментуму
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['macd'] = macd[f'MACD_{12}_{26}_{9}']
        df['macd_signal'] = macd[f'MACDs_{12}_{26}_{9}']
        df['macd_hist'] = macd[f'MACDh_{12}_{26}_{9}']

        # Додаємо Stochastic oscillator для визначення моментуму
        stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
        df['stoch_k'] = stoch[f'STOCHk_{14}_{3}_{3}']
        df['stoch_d'] = stoch[f'STOCHd_{14}_{3}_{3}']

        # 3. Аналіз поточного стану ринку

        # Видаляємо рядки з NA значеннями для коректного аналізу
        df_clean = df.dropna()

        # Якщо після видалення NA залишилось мало даних, повертаємо 'insufficient_data'
        if len(df_clean) < 20:
            return "insufficient_data"

        # Визначаємо сучасний стан індикаторів

        # Отримуємо останні значення індикаторів
        adx_value = df_clean['adx'].iloc[-1]
        plus_di = df_clean['+di'].iloc[-1]
        minus_di = df_clean['-di'].iloc[-1]

        # Середнє значення нормалізованої волатильності за останні 20 періодів
        recent_volatility = df_clean['norm_volatility'].tail(20).mean()
        historical_volatility = df_clean['norm_volatility'].mean()

        # Визначаємо квартилі волатильності
        low_vol_threshold = df_clean['norm_volatility'].quantile(0.25)
        high_vol_threshold = df_clean['norm_volatility'].quantile(0.75)

        # Аналіз ширини смуг Боллінджера за останні 10 періодів
        recent_bb_width = df_clean['bb_width'].tail(10).mean()
        historical_bb_width = df_clean['bb_width'].mean()

        # Додаткові метрики
        rsi_value = df_clean['rsi'].iloc[-1]
        macd_hist = df_clean['macd_hist'].tail(3).mean()  # Середнє значення гістограми MACD за останні 3 періоди

        # 4. Визначення режиму ринку на основі комбінації показників

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

        # Визначаємо стан волатильності
        if recent_volatility > high_vol_threshold * 1.5:
            volatility_state = "extremely_high"
        elif recent_volatility > high_vol_threshold:
            volatility_state = "high"
        elif recent_volatility < low_vol_threshold:
            volatility_state = "low"
        else:
            volatility_state = "normal"

        # Визначаємо стан ширини смуг Боллінджера (консолідація чи розширення)
        if recent_bb_width < historical_bb_width * 0.7:
            bollinger_state = "tight"  # Значна консолідація
        elif recent_bb_width < historical_bb_width * 0.9:
            bollinger_state = "narrowing"  # Звуження
        elif recent_bb_width > historical_bb_width * 1.3:
            bollinger_state = "wide"  # Значне розширення
        elif recent_bb_width > historical_bb_width * 1.1:
            bollinger_state = "expanding"  # Розширення
        else:
            bollinger_state = "normal"  # Нормальний стан

        # 5. Інтегрована оцінка режиму ринку

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

    def detect_divergence(self, price_data: pd.DataFrame, indicator_data: pd.DataFrame) -> List[Dict]:

        # Перевірка наявності необхідних даних
        if 'close' not in price_data.columns:
            raise ValueError("price_data повинен містити стовпець 'close'")

        if indicator_data.empty or len(indicator_data.columns) == 0:
            raise ValueError("indicator_data не містить даних")

        # Використовуємо перший стовпець indicator_data як індикатор, якщо не вказано інакше
        indicator_column = indicator_data.columns[0]

        # Переконуємося, що індекси співпадають
        if not price_data.index.equals(indicator_data.index):
            # Можливо потрібно переіндексувати або об'єднати дані
            common_index = price_data.index.intersection(indicator_data.index)
            price_data = price_data.loc[common_index]
            indicator_data = indicator_data.loc[common_index]

        # Дивергенції будемо шукати на локальних максимумах і мінімумах
        # Для цього знайдемо локальні екстремуми
        window = 5  # Розмір вікна для пошуку екстремумів

        # Функція для знаходження локальних максимумів
        def find_local_maxima(series, window):
            max_indices = []
            for i in range(window, len(series) - window):
                if all(series.iloc[i] > series.iloc[i - j] for j in range(1, window + 1)) and \
                        all(series.iloc[i] > series.iloc[i + j] for j in range(1, window + 1)):
                    max_indices.append(i)
            return max_indices

        # Функція для знаходження локальних мінімумів
        def find_local_minima(series, window):
            min_indices = []
            for i in range(window, len(series) - window):
                if all(series.iloc[i] < series.iloc[i - j] for j in range(1, window + 1)) and \
                        all(series.iloc[i] < series.iloc[i + j] for j in range(1, window + 1)):
                    min_indices.append(i)
            return min_indices

        # Знаходимо локальні максимуми і мінімуми для ціни і індикатора
        price_maxima = find_local_maxima(price_data['close'], window)
        price_minima = find_local_minima(price_data['close'], window)
        indicator_maxima = find_local_maxima(indicator_data[indicator_column], window)
        indicator_minima = find_local_minima(indicator_data[indicator_column], window)

        # Список для зберігання знайдених дивергенцій
        divergences = []

        # Максимальна відстань між екстремумами для пошуку дивергенцій (у періодах)
        max_distance = 3

        # Шукаємо регулярні (негативні) дивергенції на максимумах
        # Ціна росте, але індикатор не підтверджує (робить нижчий максимум)
        for i in range(1, len(price_maxima)):
            price_idx1 = price_maxima[i - 1]
            price_idx2 = price_maxima[i]

            # Шукаємо відповідні максимуми індикатора
            closest_indicator_idx1 = None
            closest_indicator_idx2 = None
            min_distance1 = max_distance
            min_distance2 = max_distance

            for idx in indicator_maxima:
                distance1 = abs(idx - price_idx1)
                distance2 = abs(idx - price_idx2)

                if distance1 < min_distance1:
                    closest_indicator_idx1 = idx
                    min_distance1 = distance1

                if distance2 < min_distance2:
                    closest_indicator_idx2 = idx
                    min_distance2 = distance2

            if closest_indicator_idx1 is not None and closest_indicator_idx2 is not None:
                # Перевіряємо наявність дивергенції (ціна росте, індикатор падає)
                price_higher = price_data['close'].iloc[price_idx2] > price_data['close'].iloc[price_idx1]
                indicator_lower = indicator_data[indicator_column].iloc[closest_indicator_idx2] < \
                                  indicator_data[indicator_column].iloc[closest_indicator_idx1]

                if price_higher and indicator_lower:
                    divergences.append({
                        'type': 'regular_bearish',
                        'price_date1': price_data.index[price_idx1],
                        'price_date2': price_data.index[price_idx2],
                        'price_value1': price_data['close'].iloc[price_idx1],
                        'price_value2': price_data['close'].iloc[price_idx2],
                        'indicator_value1': indicator_data[indicator_column].iloc[closest_indicator_idx1],
                        'indicator_value2': indicator_data[indicator_column].iloc[closest_indicator_idx2],
                        'description': 'Негативна дивергенція: ціна росте, але індикатор падає'
                    })

        # Шукаємо регулярні (позитивні) дивергенції на мінімумах
        # Ціна падає, але індикатор не підтверджує (робить вищий мінімум)
        for i in range(1, len(price_minima)):
            price_idx1 = price_minima[i - 1]
            price_idx2 = price_minima[i]

            # Шукаємо відповідні мінімуми індикатора
            closest_indicator_idx1 = None
            closest_indicator_idx2 = None
            min_distance1 = max_distance
            min_distance2 = max_distance

            for idx in indicator_minima:
                distance1 = abs(idx - price_idx1)
                distance2 = abs(idx - price_idx2)

                if distance1 < min_distance1:
                    closest_indicator_idx1 = idx
                    min_distance1 = distance1

                if distance2 < min_distance2:
                    closest_indicator_idx2 = idx
                    min_distance2 = distance2

            if closest_indicator_idx1 is not None and closest_indicator_idx2 is not None:
                # Перевіряємо наявність дивергенції (ціна падає, індикатор росте)
                price_lower = price_data['close'].iloc[price_idx2] < price_data['close'].iloc[price_idx1]
                indicator_higher = indicator_data[indicator_column].iloc[closest_indicator_idx2] > \
                                   indicator_data[indicator_column].iloc[closest_indicator_idx1]

                if price_lower and indicator_higher:
                    divergences.append({
                        'type': 'regular_bullish',
                        'price_date1': price_data.index[price_idx1],
                        'price_date2': price_data.index[price_idx2],
                        'price_value1': price_data['close'].iloc[price_idx1],
                        'price_value2': price_data['close'].iloc[price_idx2],
                        'indicator_value1': indicator_data[indicator_column].iloc[closest_indicator_idx1],
                        'indicator_value2': indicator_data[indicator_column].iloc[closest_indicator_idx2],
                        'description': 'Позитивна дивергенція: ціна падає, але індикатор росте'
                    })

        # Шукаємо приховані дивергенції на максимумах
        # Ціна робить нижчий максимум, але індикатор робить вищий максимум
        for i in range(1, len(price_maxima)):
            price_idx1 = price_maxima[i - 1]
            price_idx2 = price_maxima[i]

            # Шукаємо відповідні максимуми індикатора
            closest_indicator_idx1 = None
            closest_indicator_idx2 = None
            min_distance1 = max_distance
            min_distance2 = max_distance

            for idx in indicator_maxima:
                distance1 = abs(idx - price_idx1)
                distance2 = abs(idx - price_idx2)

                if distance1 < min_distance1:
                    closest_indicator_idx1 = idx
                    min_distance1 = distance1

                if distance2 < min_distance2:
                    closest_indicator_idx2 = idx
                    min_distance2 = distance2

            if closest_indicator_idx1 is not None and closest_indicator_idx2 is not None:
                # Перевіряємо наявність прихованої дивергенції (ціна нижче, індикатор вище)
                price_lower = price_data['close'].iloc[price_idx2] < price_data['close'].iloc[price_idx1]
                indicator_higher = indicator_data[indicator_column].iloc[closest_indicator_idx2] > \
                                   indicator_data[indicator_column].iloc[closest_indicator_idx1]

                if price_lower and indicator_higher:
                    divergences.append({
                        'type': 'hidden_bullish',
                        'price_date1': price_data.index[price_idx1],
                        'price_date2': price_data.index[price_idx2],
                        'price_value1': price_data['close'].iloc[price_idx1],
                        'price_value2': price_data['close'].iloc[price_idx2],
                        'indicator_value1': indicator_data[indicator_column].iloc[closest_indicator_idx1],
                        'indicator_value2': indicator_data[indicator_column].iloc[closest_indicator_idx2],
                        'description': 'Прихована бичача дивергенція: ціна робить нижчий максимум, але індикатор вищий максимум'
                    })

        # Шукаємо приховані дивергенції на мінімумах
        # Ціна робить вищий мінімум, але індикатор робить нижчий мінімум
        for i in range(1, len(price_minima)):
            price_idx1 = price_minima[i - 1]
            price_idx2 = price_minima[i]

            # Шукаємо відповідні мінімуми індикатора
            closest_indicator_idx1 = None
            closest_indicator_idx2 = None
            min_distance1 = max_distance
            min_distance2 = max_distance

            for idx in indicator_minima:
                distance1 = abs(idx - price_idx1)
                distance2 = abs(idx - price_idx2)

                if distance1 < min_distance1:
                    closest_indicator_idx1 = idx
                    min_distance1 = distance1

                if distance2 < min_distance2:
                    closest_indicator_idx2 = idx
                    min_distance2 = distance2

            if closest_indicator_idx1 is not None and closest_indicator_idx2 is not None:
                # Перевіряємо наявність прихованої дивергенції (ціна вище, індикатор нижче)
                price_higher = price_data['close'].iloc[price_idx2] > price_data['close'].iloc[price_idx1]
                indicator_lower = indicator_data[indicator_column].iloc[closest_indicator_idx2] < \
                                  indicator_data[indicator_column].iloc[closest_indicator_idx1]

                if price_higher and indicator_lower:
                    divergences.append({
                        'type': 'hidden_bearish',
                        'price_date1': price_data.index[price_idx1],
                        'price_date2': price_data.index[price_idx2],
                        'price_value1': price_data['close'].iloc[price_idx1],
                        'price_value2': price_data['close'].iloc[price_idx2],
                        'indicator_value1': indicator_data[indicator_column].iloc[closest_indicator_idx1],
                        'indicator_value2': indicator_data[indicator_column].iloc[closest_indicator_idx2],
                        'description': 'Прихована ведмежа дивергенція: ціна робить вищий мінімум, але індикатор нижчий мінімум'
                    })

        return divergences

    def calculate_trend_metrics(self, data: pd.DataFrame) -> Dict[str, float]:

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

    def save_trend_analysis_to_db(self, symbol: str, timeframe: str,
                                  analysis_results: Dict) -> bool:

        try:
            # Використовуємо поточну дату для аналізу
            analysis_date = datetime.now()

            # Викликаємо метод save_trend_analysis для збереження даних
            return self.db_manager.save_trend_analysis(
                symbol=symbol,
                timeframe=timeframe,
                analysis_date=analysis_date,
                trend_data=analysis_results
            )
        except Exception as e:
            self.logger.error(f"Error saving trend analysis to DB: {str(e)}")
            return False

    def load_trend_analysis_from_db(self,
            symbol: str,
            timeframe: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            latest_only: bool = False
            ) -> Dict[str, Any]:

        try:
            result = self.db_manager.get_trend_analysis(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                latest_only=latest_only
            )

            if result is None:
                return {
                    'status': 'no_data',
                    'message': 'Дані трендового аналізу не знайдено'
                }

            return {
                'status': 'success',
                'data': result
            }

        except Exception as e:
            self.logger.error(f"Помилка у get(): {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_trend_summary(self, symbol: str, timeframe: str) -> Dict:

        try:
            # Завантажуємо останній аналіз тренду з бази даних
            latest_analysis = self.load_trend_analysis_from_db(symbol, timeframe)

            if not latest_analysis:
                return {
                    'status': 'no_data',
                    'message': 'No trend analysis available for this symbol and timeframe'
                }

            # Беремо останній запис
            latest = latest_analysis[0]

            # Формуємо зведений звіт
            summary = {
                'symbol': symbol,
                'timeframe': timeframe,
                'trend_type': latest.get('trend_type', 'unknown'),
                'trend_strength': latest.get('trend_strength', 0),
                'market_regime': latest.get('market_regime', 'unknown'),
                'support_levels': latest.get('support_levels', []),
                'resistance_levels': latest.get('resistance_levels', []),
                'patterns_detected': len(latest.get('detected_patterns', [])),
                'last_analysis_date': latest.get('analysis_date'),
                'additional_metrics': latest.get('additional_metrics', {})
            }

            # Додаємо індикатори сили тренду, якщо вони є
            if 'additional_metrics' in latest:
                metrics = latest['additional_metrics']
                summary.update({
                    'trend_speed': metrics.get('speed_20', 0),
                    'trend_acceleration': metrics.get('acceleration_20', 0),
                    'volatility': metrics.get('volatility_20', 0)
                })

            return summary

        except Exception as e:
            self.logger.error(f"Error generating trend summary: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def prepare_ml_trend_features(self, data: pd.DataFrame, lookback_window: int = 30) -> pd.DataFrame:

        try:
            # Копіюємо дані щоб уникнути попереджень
            df = data.copy()

            # 1. Розраховуємо технічні індикатори
            df = self.calculate_adx(df)  # ADX, +DI, -DI
            df['rsi'] = ta.rsi(df['close'])  # RSI
            macd = ta.macd(df['close'])  # MACD
            df = pd.concat([df, macd], axis=1)

            # 2. Додаємо метрики тренду
            metrics = self.calculate_trend_metrics(df)
            for k, v in metrics.items():
                if v is not None:
                    df[k] = v

            # 3. Додаємо інші важливі ознаки
            df['trend_strength'] = self.calculate_trend_strength(df)
            df['market_regime'] = self.identify_market_regime(df)  # Можна закодувати як число

            # 4. Нормалізація даних (важливо для LSTM/GRU)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            feature_cols = ['close', 'volume', 'adx', '+di', '-di', 'rsi',
                            'MACD_12_26_9', 'MACDs_12_26_9', 'trend_strength',
                            'speed_20', 'volatility_20']

            # Залишаємо тільки потрібні колонки
            df = df[feature_cols + ['market_regime']]

            # Нормалізуємо числові ознаки
            df[feature_cols] = scaler.fit_transform(df[feature_cols])

            # 5. Створюємо часові вікна для LSTM/GRU
            X, y = [], []
            for i in range(lookback_window, len(df)):
                X.append(df[feature_cols].values[i - lookback_window:i])
                y.append(df['close'].values[i])  # Можна змінити на прогноз зміни ціни

            # 6. Повертаємо дані у вигляді numpy масивів
            return np.array(X), np.array(y), df['market_regime'].values[lookback_window:]

        except Exception as e:
            self.logger.error(f"Error preparing ML features: {str(e)}")
            return None, None, None

def main():
    # Initialize TrendDetection
    trend_detector = TrendDetection()

    # Get real market data from exchange
    symbol = "BTC"
    timeframe = "1d"

    try:
        # Fetch klines data (assuming db_manager has get_klines method)
        klines = trend_detector.db_manager.get_klines(
            symbol=symbol,
            timeframe=timeframe,
        )

        # Convert to DataFrame
        test_data = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Convert columns to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        test_data[numeric_cols] = test_data[numeric_cols].apply(pd.to_numeric)

        # Convert timestamp to datetime
        test_data['date'] = pd.to_datetime(test_data['open_time'], unit='ms')
        test_data = test_data.set_index('date')

        print(f"\nFetched {len(test_data)} candles for {symbol} {timeframe}")
        print("Latest data points:")
        print(test_data[['open', 'high', 'low', 'close']])

        # Test trend detection
        trend = trend_detector.detect_trend(test_data)
        print("\n1. Detected trend:", trend)

        # Test support/resistance levels
        sr_levels = trend_detector.identify_support_resistance(test_data)
        print("\n2. Support levels:", [round(x, 2) for x in sr_levels['support'][-3:]])
        print("   Resistance levels:", [round(x, 2) for x in sr_levels['resistance'][-3:]])

        # Test breakouts
        breakouts = trend_detector.detect_breakouts(test_data, sr_levels)
        print("\n3. Detected breakouts:", len(breakouts))
        if breakouts:
            latest_breakout = breakouts[-1]
            print(
                f"   Latest breakout: {latest_breakout['type']} at {latest_breakout['level']:.2f} on {latest_breakout['date']}")

        # Test trend strength
        strength = trend_detector.calculate_trend_strength(test_data)
        print("\n4. Trend strength (0-1):", round(strength, 2))

        # Test trend reversal detection
        reversals = trend_detector.detect_trend_reversal(test_data)
        print("\n5. Detected reversals:", len(reversals))
        if reversals:
            latest_reversal = reversals[-1]
            print(f"   Latest reversal: {latest_reversal['signal_type']} on {latest_reversal['date']}")

        # Test Fibonacci levels
        fib_levels = trend_detector.calculate_fibonacci_levels(test_data, trend)
        print("\n6. Fibonacci retracement levels:")
        for level, price in sorted(fib_levels.items()):
            if level not in ['swing_high', 'swing_low']:
                print(f"   {level}: {price:.2f}")

        # Test chart patterns
        patterns = trend_detector.detect_chart_patterns(test_data)
        print("\n7. Detected chart patterns:", len(patterns))
        if patterns:
            latest_pattern = patterns[-1]
            print(
                f"   Latest pattern: {latest_pattern['type']} from {latest_pattern['start_date']} to {latest_pattern['end_date']}")

        # Test market regime
        regime = trend_detector.identify_market_regime(test_data)
        print("\n8. Current market regime:", regime)

        # Test trend metrics
        metrics = trend_detector.calculate_trend_metrics(test_data)
        print("\n9. Key trend metrics:")
        print(f"   Speed (20): {metrics['speed_20']:.4f}")
        print(f"   Acceleration (20): {metrics['acceleration_20']:.4f}")
        print(f"   Volatility (20): {metrics['volatility_20']:.4f}")
        print(f"   Trend strength: {metrics['trend_strength']:.2f}")

        # Save results to DB
        analysis_results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'trend_type': trend,
            'trend_strength': strength,
            'support_levels': sr_levels['support'],
            'resistance_levels': sr_levels['resistance'],
            'market_regime': regime,
            'detected_patterns': [p['type'] for p in patterns],
            'analysis_date': datetime.now().isoformat(),
            'additional_metrics': metrics
        }

        save_result = trend_detector.save_trend_analysis_to_db(
            symbol=symbol,
            timeframe=timeframe,
            analysis_results=analysis_results
        )
        print("\n10. Save to DB successful:", save_result)

        # Get trend summary
        summary = trend_detector.get_trend_summary(symbol, timeframe)
        print("\n11. Trend Summary:")
        print(f"   Symbol: {summary.get('symbol')}")
        print(f"   Trend: {summary.get('trend_type')}")
        print(f"   Strength: {summary.get('trend_strength', 0):.2f}")
        print(f"   Market Regime: {summary.get('market_regime')}")
        print(f"   Support Levels: {len(summary.get('support_levels', []))}")
        print(f"   Resistance Levels: {len(summary.get('resistance_levels', []))}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")


if __name__ == "__main__":
    main()