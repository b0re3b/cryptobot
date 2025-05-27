from typing import Dict, List
import pandas as pd
from utils.logger import CryptoLogger

class SupportResistantAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = CryptoLogger('supportresistantanalyzer')
        self.min_points_for_level = self.config.get('min_points_for_level', 3)

    def _group_price_levels(self, price_points: List[float], threshold: float) -> List[float]:
        """Групування близьких цінових рівнів"""
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

    def identify_support_resistance(self, data: pd.DataFrame,
                                    window_size: int = 20,
                                    threshold: float = 0.02) -> Dict[str, List[float]]:
        """Ідентифікація рівнів підтримки та опору"""
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

            # Перевірка достатності даних
            if len(data) < window_size * 2 + 1:
                self.logger.warning(f"Insufficient data for window_size {window_size}")
                return {"support": [], "resistance": []}

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

    def detect_breakouts(self, data: pd.DataFrame,
                         support_resistance: Dict[str, List[float]],
                         threshold: float = 0.01) -> List[Dict]:
        """Детекція пробоїв рівнів підтримки та опору"""
        if data.empty or not support_resistance:
            return []

        # Переконаємося, що необхідні колонки присутні
        required_columns = ['close', 'high', 'low']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Дані повинні містити колонки 'close', 'high', 'low'")

        # Створення копії даних для уникнення змін оригіналу
        data_copy = data.copy()

        # Якщо немає колонки 'date', використаємо індекс
        if 'date' not in data_copy.columns:
            if isinstance(data_copy.index, pd.DatetimeIndex):
                data_copy = data_copy.reset_index()
                data_copy.rename(columns={'index': 'date'}, inplace=True)
            else:
                data_copy['date'] = data_copy.index

        breakouts = []

        # Отримуємо рівні підтримки та опору
        support_levels = sorted(support_resistance.get('support', []))
        resistance_levels = sorted(support_resistance.get('resistance', []))

        # Перевіряємо пробій кожного рівня опору
        for level in resistance_levels:
            # Знаходимо дні, коли ціна закриття була нижче рівня, а потім пробила вгору
            for i in range(1, len(data_copy)):
                # Перевіряємо, чи ціна закриття була нижче рівня, а потім пробила його вище порогу
                if (data_copy['close'].iloc[i - 1] < level and
                        data_copy['close'].iloc[i] > level * (1 + threshold)):

                    breakout = {
                        'type': 'resistance_breakout',
                        'level': level,
                        'date': data_copy['date'].iloc[i],
                        'price': data_copy['close'].iloc[i],
                        'strength': (data_copy['close'].iloc[i] - level) / level,
                        'volume_change': None
                    }

                    # Додаємо зміну об'єму, якщо дані доступні
                    if 'volume' in data_copy.columns and i > 0:
                        if data_copy['volume'].iloc[i - 1] != 0:  # Уникаємо ділення на нуль
                            volume_change = (data_copy['volume'].iloc[i] / data_copy['volume'].iloc[i - 1]) - 1
                            breakout['volume_change'] = volume_change

                    breakouts.append(breakout)

        # Перевіряємо пробій кожного рівня підтримки
        for level in support_levels:
            # Знаходимо дні, коли ціна закриття була вище рівня, а потім пробила вниз
            for i in range(1, len(data_copy)):
                # Перевіряємо, чи ціна закриття була вище рівня, а потім пробила його нижче порогу
                if (data_copy['close'].iloc[i - 1] > level and
                        data_copy['close'].iloc[i] < level * (1 - threshold)):

                    breakout = {
                        'type': 'support_breakout',
                        'level': level,
                        'date': data_copy['date'].iloc[i],
                        'price': data_copy['close'].iloc[i],
                        'strength': (level - data_copy['close'].iloc[i]) / level,
                        'volume_change': None
                    }

                    # Додаємо зміну об'єму, якщо дані доступні
                    if 'volume' in data_copy.columns and i > 0:
                        if data_copy['volume'].iloc[i - 1] != 0:  # Уникаємо ділення на нуль
                            volume_change = (data_copy['volume'].iloc[i] / data_copy['volume'].iloc[i - 1]) - 1
                            breakout['volume_change'] = volume_change

                    breakouts.append(breakout)

        # Сортуємо пробої за датою
        breakouts = sorted(breakouts, key=lambda x: x['date'])

        return breakouts

    def find_swing_points(self, data: pd.DataFrame, window_size: int = 5) -> Dict[str, List[Dict]]:
        """Знаходження поворотних точок (swing points)"""
        if data.empty or len(data) < window_size * 2 + 1:
            return {'highs': [], 'lows': []}

        # Перевіряємо наявність необхідних колонок
        required_columns = ['high', 'low']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Дані повинні містити колонки 'high' та 'low'")

        # Створення копії даних
        data_copy = data.copy()

        # Якщо немає колонки 'date', використаємо індекс
        if 'date' not in data_copy.columns:
            if isinstance(data_copy.index, pd.DatetimeIndex):
                data_copy = data_copy.reset_index()
                data_copy.rename(columns={'index': 'date'}, inplace=True)
            else:
                data_copy['date'] = data_copy.index

        swing_points = {'highs': [], 'lows': []}

        # Для кожної точки перевіряємо, чи є вона локальним екстремумом
        for i in range(window_size, len(data_copy) - window_size):
            # Перевіряємо, чи є точка локальним максимумом
            is_high = True
            for j in range(1, window_size + 1):
                if (data_copy['high'].iloc[i] <= data_copy['high'].iloc[i - j] or
                        data_copy['high'].iloc[i] <= data_copy['high'].iloc[i + j]):
                    is_high = False
                    break

            if is_high:
                # Розраховуємо силу точки (наскільки вона виділяється)
                left_diffs = [data_copy['high'].iloc[i] - data_copy['high'].iloc[i - j]
                              for j in range(1, window_size + 1)]
                right_diffs = [data_copy['high'].iloc[i] - data_copy['high'].iloc[i + j]
                               for j in range(1, window_size + 1)]

                swing_high = {
                    'date': data_copy['date'].iloc[i],
                    'price': data_copy['high'].iloc[i],
                    'index': i,
                    'strength': min(min(left_diffs), min(right_diffs)) / data_copy['high'].iloc[i]
                }

                # Додаємо додаткові дані, якщо доступні
                if 'volume' in data_copy.columns:
                    swing_high['volume'] = data_copy['volume'].iloc[i]

                swing_points['highs'].append(swing_high)

            # Перевіряємо, чи є точка локальним мінімумом
            is_low = True
            for j in range(1, window_size + 1):
                if (data_copy['low'].iloc[i] >= data_copy['low'].iloc[i - j] or
                        data_copy['low'].iloc[i] >= data_copy['low'].iloc[i + j]):
                    is_low = False
                    break

            if is_low:
                # Розраховуємо силу точки (наскільки вона виділяється)
                left_diffs = [data_copy['low'].iloc[i - j] - data_copy['low'].iloc[i]
                              for j in range(1, window_size + 1)]
                right_diffs = [data_copy['low'].iloc[i + j] - data_copy['low'].iloc[i]
                               for j in range(1, window_size + 1)]

                swing_low = {
                    'date': data_copy['date'].iloc[i],
                    'price': data_copy['low'].iloc[i],
                    'index': i,
                    'strength': min(min(left_diffs), min(right_diffs)) / data_copy['low'].iloc[i]
                }

                # Додаємо додаткові дані, якщо доступні
                if 'volume' in data_copy.columns:
                    swing_low['volume'] = data_copy['volume'].iloc[i]

                swing_points['lows'].append(swing_low)

        return swing_points

    def calculate_fibonacci_levels(self, data: pd.DataFrame, trend_type: str) -> Dict[str, float]:
        """Розрахунок рівнів Фібоначчі"""
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

        # Рівні ретрейсменту та екстенжн
        retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        extension_levels = [1.272, 1.618, 2.618]

        # Розрахунок цінових рівнів на основі відрізка [start_price, end_price]
        price_range = end_price - start_price

        result = {}
        # Додаємо рівні ретрейсменту
        for level in retracement_levels:
            level_str = f"{level:.3f}" if level not in [0, 1] else f"{level:.1f}"
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