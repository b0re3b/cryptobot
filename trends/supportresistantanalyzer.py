from typing import Dict, List
import pandas as pd

from utils.logger import CryptoLogger


class SupportResistantAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = CryptoLogger('supportresistantanalyzer')
        self.min_points_for_level = self.config.get('min_points_for_level', 3)

    def _group_price_levels(self, price_points: List[float], threshold: float) -> List[float]:
        """
    Групує близькі між собою цінові рівні на основі відносного порогу.

    Метод об'єднує точки ціни, що знаходяться в межах `threshold` (відносної різниці) одна від одної,
    у групи. Для кожної валідної групи (з кількістю точок >= self.min_points_for_level)
    розраховується середнє значення, яке представляє ціновий рівень.

    Args:
        price_points (List[float]): Список цінових точок (float), які потрібно згрупувати.
        threshold (float): Відносний поріг для об'єднання точок в одну групу (наприклад, 0.01 = 1%).

    Returns:
        List[float]: Список згрупованих цінових рівнів, представлених як середнє значення кожної групи.

    Notes:
        - Групування базується на першій точці групи: всі наступні точки вважаються "близькими",
          якщо їхня відносна різниця з першою точкою групи не перевищує threshold.
        - Групи з кількістю точок менше `self.min_points_for_level` ігноруються.
        - Вихідний список сортується перед обробкою.


    """
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
        """
            Ідентифікує рівні підтримки та опору на основі локальних екстремумів ціни.

            Для кожної точки на графіку перевіряється, чи є вона локальним мінімумом або максимумом у заданому
            `window_size` контексті. Знайдені рівні групуються за відносною близькістю (`threshold`)
            для уникнення дублювання.

            Args:
                data (pd.DataFrame): Історичні цінові дані з колонками 'high' і 'low' або 'close'.
                window_size (int, optional): Кількість попередніх та наступних точок для перевірки на локальний екстремум.
                    Має бути достатньо великим для згладжування шумів, типово 20.
                threshold (float, optional): Відносний поріг (наприклад, 0.02 = 2%) для групування близьких рівнів.

            Returns:
                Dict[str, List[float]]: Словник з двома ключами:
                    - "support": список згрупованих рівнів підтримки (локальні мінімуми)
                    - "resistance": список згрупованих рівнів опору (локальні максимуми)

            Raises:
                ValueError: Якщо у вхідному DataFrame відсутні необхідні стовпці ('high', 'low' або 'close').

            Notes:
                - Якщо немає 'high' і 'low', але є 'close', метод використовує останню для обох.
                - Для коректної роботи потрібно принаймні `(2 * window_size + 1)` рядків.
                - Групування екстремумів виконується методом `_group_price_levels`, який об’єднує близькі ціни.


            """
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
            # Знаходимо дні, коли ціна була нижче рівня, а потім пробила вгору
            for i in range(1, len(data_copy)):
                current_close = data_copy['close'].iloc[i]
                previous_close = data_copy['close'].iloc[i - 1]
                current_high = data_copy['high'].iloc[i]

                # Пробій опору: попередня ціна була нижче рівня, поточна high пробила рівень
                # і поточна close закрилася вище рівня з урахуванням порогу
                if (previous_close < level and
                        current_high > level and
                        current_close > level * (1 + threshold)):

                    breakout = {
                        'type': 'resistance_breakout',
                        'level': level,
                        'date': data_copy['date'].iloc[i],
                        'price': current_close,
                        'high_price': current_high,  # Додаємо максимальну ціну дня
                        'strength': (current_close - level) / level,
                        'volume_change': None
                    }

                    # Додаємо зміну об'єму, якщо дані доступні
                    if 'volume' in data_copy.columns and i > 0:
                        prev_volume = data_copy['volume'].iloc[i - 1]
                        curr_volume = data_copy['volume'].iloc[i]
                        if prev_volume > 0:  # Уникаємо ділення на нуль
                            volume_change = (curr_volume / prev_volume) - 1
                            breakout['volume_change'] = volume_change

                    breakouts.append(breakout)

        # Перевіряємо пробій кожного рівня підтримки
        for level in support_levels:
            # Знаходимо дні, коли ціна була вище рівня, а потім пробила вниз
            for i in range(1, len(data_copy)):
                current_close = data_copy['close'].iloc[i]
                previous_close = data_copy['close'].iloc[i - 1]
                current_low = data_copy['low'].iloc[i]

                # Пробій підтримки: попередня ціна була вище рівня, поточна low пробила рівень
                # і поточна close закрилася нижче рівня з урахуванням порогу
                if (previous_close > level and
                        current_low < level and
                        current_close < level * (1 - threshold)):

                    breakout = {
                        'type': 'support_breakout',
                        'level': level,
                        'date': data_copy['date'].iloc[i],
                        'price': current_close,
                        'low_price': current_low,  # Додаємо мінімальну ціну дня
                        'strength': (level - current_close) / level,
                        'volume_change': None
                    }

                    # Додаємо зміну об'єму, якщо дані доступні
                    if 'volume' in data_copy.columns and i > 0:
                        prev_volume = data_copy['volume'].iloc[i - 1]
                        curr_volume = data_copy['volume'].iloc[i]
                        if prev_volume > 0:  # Уникаємо ділення на нуль
                            volume_change = (curr_volume / prev_volume) - 1
                            breakout['volume_change'] = volume_change

                    breakouts.append(breakout)

        # Сортуємо пробої за датою
        breakouts = sorted(breakouts, key=lambda x: x['date'])

        return breakouts

    def find_swing_points(self, data: pd.DataFrame, window_size: int = 5) -> Dict[str, List[Dict]]:
        """
            Визначає локальні поворотні точки (swing highs та swing lows) у ціновому ряді.

            Метод аналізує кожну точку і перевіряє, чи вона є локальним максимумом або мінімумом
            в заданому `window_size`. Для кожної swing-точки обчислюється сила (strength) — наскільки
            ціна виділяється серед сусідніх значень.

            Args:
                data (pd.DataFrame): OHLC[V]-дані з колонками 'high', 'low'.
                    Опціонально може містити 'volume' для додаткової інформації.
                window_size (int): Кількість барів зліва і справа для перевірки локального екстремуму.

            Returns:
                Dict[str, List[Dict]]: Словник з ключами:
                    - 'highs': список swing high точок, кожна у вигляді словника:
                        {
                            'date': дата точки,
                            'price': значення high,
                            'index': індекс у DataFrame,
                            'strength': відносна сила (відхилення від найближчих значень),
                            'volume': (опційно) обʼєм у цій точці
                        }
                    - 'lows': список swing low точок, з такими ж полями

            Raises:
                ValueError: Якщо вхідні дані не містять обов’язкових колонок 'high' і 'low'.

            Notes:
                - Точка вважається swing high, якщо її high більше, ніж у `window_size` сусідів зліва і справа.
                - Аналогічно для swing low.
                - Сила обчислюється як мінімальне відхилення від сусідніх значень (зліва і справа), нормалізоване до самої ціни.


            """
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

    def calculate_fibonacci_levels(self, data: pd.DataFrame, trend_type: str,
                                   lookback_period: int = 30) -> Dict[str, float]:
        """
            Розраховує ключові рівні Фібоначчі (ретрейсменту та розширення) на основі останнього тренду.

            Метод визначає локальні swing high/low у вказаному lookback-вікні та обчислює
            рівні Фібоначчі відповідно до типу тренду (висхідний або низхідний).
            Додатково повертає метаінформацію: діапазон, поточну ціну, swing-точки та найближчі рівні підтримки/опору.

            Args:
                data (pd.DataFrame): OHLC[V]-дані з колонками 'high', 'low', 'close'.
                trend_type (str): Тип тренду: 'uptrend' або 'downtrend'.
                lookback_period (int): Кількість останніх періодів, які враховуються при аналізі тренду.

            Returns:
                Dict[str, float]: Словник із розрахованими рівнями Фібоначчі та метаданими:
                    - Рівні ретрейсменту: 'fib_0.236_retracement', ..., 'fib_1.0_retracement'
                    - Рівні розширення: 'fib_1.272_extension', ..., 'fib_2.618_extension'
                    - Метаінформація:
                        - 'swing_high': значення swing high
                        - 'swing_low': значення swing low
                        - 'price_range': різниця між high і low
                        - 'current_price': остання ціна закриття
                        - 'trend_type': вхідний параметр
                        - 'lookback_period_used': фактична кількість використаних рядків
                        - 'closest_support': найближчий рівень нижче поточної ціни
                        - 'closest_resistance': найближчий рівень вище поточної ціни

            Raises:
                ValueError: Якщо дані не містять необхідних колонок або тип тренду некоректний.

            Notes:
                - Для 'uptrend': рівні ретрейсменту розраховуються від swing high вниз.
                - Для 'downtrend': рівні ретрейсменту розраховуються від swing low вгору.
                - Рівні розширення будуються за межами swing-точок у напрямку тренду.
                - Якщо ціна майже не змінилася (менше ніж 0.5%), повертається порожній словник.
            """
        if data.empty or len(data) < 2:
            self.logger.warning("Недостатньо даних для розрахунку рівнів Фібоначчі")
            return {}

        # Перевіряємо наявність необхідних колонок
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {', '.join(required_columns)}")

        # Перевіряємо тип тренду
        if trend_type not in ['uptrend', 'downtrend']:
            raise ValueError("trend_type повинен бути 'uptrend' або 'downtrend'")

        # Використовуємо останні дані замість всього діапазону
        window = min(lookback_period, len(data))
        recent_data = data.tail(window).copy()

        self.logger.info(f"Використовуємо останні {len(recent_data)} записів для розрахунку Фібоначчі")

        # Знаходимо значущі swing points у останніх даних
        if trend_type == 'uptrend':
            # Для висхідного тренду знаходимо останній значущий мінімум та максимум після нього
            swing_low_idx = recent_data['low'].idxmin()
            swing_low_position = recent_data.index.get_loc(swing_low_idx)

            # Шукаємо максимум після цього мінімуму
            data_after_low = recent_data.iloc[swing_low_position:]
            if len(data_after_low) > 1:
                swing_high_idx = data_after_low['high'].idxmax()
                start_price = recent_data.loc[swing_low_idx, 'low']  # Swing Low
                end_price = data_after_low.loc[swing_high_idx, 'high']  # Swing High
            else:
                # Fallback: використовуємо глобальні екстремуми
                start_price = recent_data['low'].min()
                end_price = recent_data['high'].max()

        else:  # downtrend
            # Для низхідного тренду знаходимо останній значущий максимум та мінімум після нього
            swing_high_idx = recent_data['high'].idxmax()
            swing_high_position = recent_data.index.get_loc(swing_high_idx)

            # Шукаємо мінімум після цього максимуму
            data_after_high = recent_data.iloc[swing_high_position:]
            if len(data_after_high) > 1:
                swing_low_idx = data_after_high['low'].idxmin()
                start_price = recent_data.loc[swing_high_idx, 'high']  # Swing High
                end_price = data_after_high.loc[swing_low_idx, 'low']  # Swing Low
            else:
                # Fallback: використовуємо глобальні екстремуми
                start_price = recent_data['high'].max()
                end_price = recent_data['low'].min()

        # Перевіряємо значущість ціного діапазону
        price_range = abs(end_price - start_price)
        min_significant_range = start_price * 0.005  # Мінімум 0.5% зміни

        if price_range < min_significant_range:
            self.logger.warning(f"Ціновий діапазон {price_range:.2f} занадто малий для значущих рівнів Фібоначчі")
            return {}

        # Рівні ретрейсменту Фібоначчі
        retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        # Рівні розширення Фібоначчі
        extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618]

        result = {}
        current_price = recent_data['close'].iloc[-1]

        # Розрахунок рівнів для висхідного тренду
        if trend_type == 'uptrend':
            # Рівні ретрейсменту: від swing high вниз до swing low
            for level in retracement_levels:
                price_level = end_price - (price_range * level)
                if level in [0, 1]:
                    level_name = f"fib_{level:.1f}_retracement"
                else:
                    level_name = f"fib_{level:.3f}_retracement"
                result[level_name] = round(price_level, 2)

            # Рівні розширення: від swing high вгору (правильний розрахунок)
            for level in extension_levels:
                price_level = end_price + (price_range * (level - 1))
                level_name = f"fib_{level:.3f}_extension"
                result[level_name] = round(price_level, 2)

        else:  # downtrend
            # Рівні ретрейсменту: від swing low вгору до swing high
            for level in retracement_levels:
                price_level = end_price + (price_range * level)
                if level in [0, 1]:
                    level_name = f"fib_{level:.1f}_retracement"
                else:
                    level_name = f"fib_{level:.3f}_retracement"
                result[level_name] = round(price_level, 2)

            # Рівні розширення: від swing low вниз (правильний розрахунок)
            for level in extension_levels:
                price_level = end_price - (price_range * (level - 1))
                level_name = f"fib_{level:.3f}_extension"
                result[level_name] = round(price_level, 2)

        # Метадані для перевірки розрахунків
        result['swing_high'] = round(max(start_price, end_price), 2)
        result['swing_low'] = round(min(start_price, end_price), 2)
        result['price_range'] = round(price_range, 2)
        result['current_price'] = round(current_price, 2)
        result['trend_type'] = trend_type
        result['lookback_period_used'] = len(recent_data)

        # Визначаємо найближчі рівні до поточної ціни
        fib_levels = {k: v for k, v in result.items() if k.startswith('fib_')}
        if fib_levels:
            closest_above = min([v for v in fib_levels.values() if v > current_price], default=None)
            closest_below = max([v for v in fib_levels.values() if v < current_price], default=None)

            result['closest_resistance'] = closest_above
            result['closest_support'] = closest_below

        self.logger.info(f"Розраховано {len(fib_levels)} рівнів Фібоначчі для {trend_type}")
        self.logger.info(f"Swing High: {result['swing_high']}, Swing Low: {result['swing_low']}")
        self.logger.info(f"Поточна ціна: {current_price:.2f}, діапазон: {price_range:.2f}")

        return result