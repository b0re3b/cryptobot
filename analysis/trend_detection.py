import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

from models.time_series import detect_seasonality
from data_collection.market_data_processor import preprocess_pipeline
from data_collection.feature_engineering import create_technical_features
from utils.logger import get_logger
from data.db import DatabaseManager


class TrendDetection:
    """
    Клас для виявлення трендів на криптовалютному ринку.
    Служить для аналізу цінових паттернів, визначення напрямку руху ринку
    та виявлення ключових рівнів підтримки/опору.
    """

    def __init__(self, config=None, logger=None):
        """
        Ініціалізація класу TrendDetection.

        Args:
            config: Конфігураційні параметри
            logger: Логер для запису подій
        """
        self.db_manager = DatabaseManager()

        # Ініціалізація конфігурації
        self.config = config or {}

        # Встановлення значень за замовчуванням, якщо не вказано інше
        self.default_window = self.config.get('default_window', 14)
        self.default_threshold = self.config.get('default_threshold', 0.02)
        self.min_points_for_level = self.config.get('min_points_for_level', 3)

        # Ініціалізація логера
        self.logger = logger or self._setup_default_logger()

    def _setup_default_logger(self) -> logging.Logger:
        """
        Створення логера за замовчуванням, якщо не передано інший.

        Returns:
            logging.Logger: Сконфігурований логер
        """
        logger = logging.getLogger('trend_detection')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def detect_trend(self, data: pd.DataFrame, window_size: int = 14) -> str:
        """
        Визначення поточного тренду (висхідний, низхідний, боковий).

        Використовує лінійну регресію для визначення нахилу цінового тренду
        та перевіряє статистичну значущість нахилу.

        Args:
            data: Дані ціни у форматі DataFrame
            window_size: Розмір вікна для аналізу тренду

        Returns:
            str: Тип тренду ('uptrend', 'downtrend', 'sideways')
        """
        try:
            # Перевірка наявності даних
            if data.empty:
                self.logger.warning("Empty DataFrame provided for trend detection")
                return "unknown"

            # Перевірка, що data містить необхідний стовпець з ціною закриття
            if 'close' not in data.columns:
                self.logger.error("DataFrame must contain 'close' column for trend detection")
                raise ValueError("DataFrame must contain 'close' column")

            # Використовуємо тільки останні window_size точок для визначення поточного тренду
            recent_data = data.tail(window_size)

            # Обчислення простого лінійного тренду
            x = np.arange(len(recent_data))
            y = recent_data['close'].values

            # Обчислення коефіцієнтів лінійної регресії (поліном 1-го ступеня)
            slope, intercept = np.polyfit(x, y, 1)

            # Обчислення відносної зміни для визначення значущості тренду
            # (нахил відносно до середньої ціни)
            mean_price = np.mean(y)
            relative_slope = slope / mean_price

            # Обчислення R-квадрат для оцінки сили тренду
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - mean_price) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Обчислення стандартного відхилення цін відносно тренду
            std_dev = np.std(y - y_pred) / mean_price

            # Логуємо деталі обчислень для діагностики
            self.logger.debug(f"Slope: {slope}, Relative Slope: {relative_slope}, R²: {r_squared}, Std Dev: {std_dev}")

            # Визначення типу тренду на основі нахилу та статистичної значущості
            # Значення порогів можна винести в конфігурацію
            slope_threshold = self.config.get('slope_threshold', 0.0005)
            r_squared_threshold = self.config.get('r_squared_threshold', 0.5)
            volatility_threshold = self.config.get('volatility_threshold', 0.02)

            # Перевірка на боковий тренд (висока волатильність або низька значущість тренду)
            if abs(relative_slope) < slope_threshold or r_squared < r_squared_threshold or std_dev > volatility_threshold:
                return "sideways"
            # Якщо нахил значущий та позитивний - висхідний тренд
            elif relative_slope > 0:
                return "uptrend"
            # Якщо нахил значущий та негативний - низхідний тренд
            else:
                return "downtrend"

        except Exception as e:
            self.logger.error(f"Error in trend detection: {str(e)}")
            return "unknown"

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Розрахунок індикатора ADX (Average Directional Index) для визначення сили тренду.

        ADX показує силу тренду незалежно від його напрямку. Значення вище 25 вказують на
        сильний тренд, нижче 20 - на відсутність тренду.

        Args:
            data: Дані OHLCV у форматі DataFrame
            period: Період для розрахунку ADX

        Returns:
            pd.DataFrame: Дані з доданими індикаторами ADX, +DI та -DI
        """
        try:
            # Перевірка наявності необхідних стовпців
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"DataFrame must contain columns: {required_columns}")
                raise ValueError(f"DataFrame must contain columns: {required_columns}")

            # Створимо копію даних, щоб не змінювати вхідний DataFrame
            result = data.copy()

            # Обчислення True Range (TR)
            result['tr0'] = abs(result['high'] - result['low'])
            result['tr1'] = abs(result['high'] - result['close'].shift(1))
            result['tr2'] = abs(result['low'] - result['close'].shift(1))
            result['tr'] = result[['tr0', 'tr1', 'tr2']].max(axis=1)

            # Обчислення напрямків руху (directional movement)
            result['up_move'] = result['high'] - result['high'].shift(1)
            result['down_move'] = result['low'].shift(1) - result['low']

            # Визначення позитивного і негативного напрямку руху
            result['plus_dm'] = np.where(
                (result['up_move'] > result['down_move']) & (result['up_move'] > 0),
                result['up_move'],
                0
            )
            result['minus_dm'] = np.where(
                (result['down_move'] > result['up_move']) & (result['down_move'] > 0),
                result['down_move'],
                0
            )

            # Розрахунок згладжених значень TR, +DM, -DM
            # Використовуємо модифікований експоненціальний фільтр Wilder
            alpha = 1 / period

            # Ініціалізація перших значень як суму за перший період
            smoothed_tr = result['tr'].rolling(window=period).sum().fillna(0)
            smoothed_plus_dm = result['plus_dm'].rolling(window=period).sum().fillna(0)
            smoothed_minus_dm = result['minus_dm'].rolling(window=period).sum().fillna(0)

            # Застосування згладжування Wilder
            for i in range(period, len(result)):
                smoothed_tr.iloc[i] = smoothed_tr.iloc[i - 1] - (smoothed_tr.iloc[i - 1] / period) + result['tr'].iloc[
                    i]
                smoothed_plus_dm.iloc[i] = smoothed_plus_dm.iloc[i - 1] - (smoothed_plus_dm.iloc[i - 1] / period) + \
                                           result['plus_dm'].iloc[i]
                smoothed_minus_dm.iloc[i] = smoothed_minus_dm.iloc[i - 1] - (smoothed_minus_dm.iloc[i - 1] / period) + \
                                            result['minus_dm'].iloc[i]

            # Розрахунок індексів напрямку +DI та -DI
            result['plus_di'] = 100 * (smoothed_plus_dm / smoothed_tr)
            result['minus_di'] = 100 * (smoothed_minus_dm / smoothed_tr)

            # Розрахунок різниці та суми індексів напрямку
            result['di_diff'] = abs(result['plus_di'] - result['minus_di'])
            result['di_sum'] = result['plus_di'] + result['minus_di']

            # Розрахунок DX (Directional Index)
            result['dx'] = 100 * (result['di_diff'] / result['di_sum'])

            # Розрахунок ADX як згладженого значення DX
            result['adx'] = result['dx'].rolling(window=period).mean()

            # Видалення проміжних обчислень
            columns_to_drop = ['tr0', 'tr1', 'tr2', 'tr', 'up_move', 'down_move',
                               'plus_dm', 'minus_dm', 'di_diff', 'di_sum', 'dx']
            result = result.drop(columns=columns_to_drop)

            # Заповнення NaN значень
            result = result.fillna(0)

            self.logger.info(f"ADX calculated successfully for period {period}")
            return result

        except Exception as e:
            self.logger.error(f"Error in ADX calculation: {str(e)}")
            # Повертаємо вхідні дані без змін у випадку помилки
            return data

    def identify_support_resistance(self, data: pd.DataFrame,
                                    window_size: int = 20,
                                    threshold: float = 0.02) -> Dict[str, List[float]]:
        """
        Визначення рівнів підтримки та опору.

        Метод визначає локальні мінімуми та максимуми,
        і групує їх для виявлення важливих цінових рівнів.

        Args:
            data: Цінові дані у форматі DataFrame
            window_size: Розмір вікна для аналізу
            threshold: Поріг для визначення значимих рівнів

        Returns:
            Dict[str, List[float]]: Словник з рівнями підтримки та опору
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
        """
        Групування близьких цінових рівнів.

        Args:
            price_points: Список цінових точок
            threshold: Відносний поріг для групування (відсоток від ціни)

        Returns:
            List[float]: Список згрупованих цінових рівнів
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

    def detect_breakouts(self, data: pd.DataFrame,
                         support_resistance: Dict[str, List[float]],
                         threshold: float = 0.01) -> List[Dict]:
        """
        Виявлення пробоїв рівнів підтримки та опору.

        Args:
            data: Цінові дані у форматі DataFrame
            support_resistance: Словник з рівнями підтримки та опору
            threshold: Поріг для визначення пробою

        Returns:
            List[Dict]: Список пробоїв з відповідними даними
        """
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
        """
        Знаходження точок розвороту (swing high/low).

        Args:
            data: Цінові дані у форматі DataFrame
            window_size: Розмір вікна для аналізу

        Returns:
            Dict[str, List[Dict]]: Словник з точками high/low та їх параметрами
        """
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

        if data.empty or len(data) < 20:  # Потрібно достатньо даних для аналізу
            return 0.0

        # Перевіряємо наявність необхідних колонок
        if 'close' not in data.columns:
            raise ValueError("Дані повинні містити колонку 'close'")

        # Розраховуємо декілька показників для визначення сили тренду

        # 1. Напрямок руху (лінійна регресія)
        x = np.arange(len(data))
        y = data['close'].values
        slope, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)[0:5]
        trend_direction = np.sign(slope)  # 1 для висхідного, -1 для низхідного
        r_squared = r_value ** 2  # Коефіцієнт детермінації (0-1)

        # 2. Аналіз консистентності тренду (відсоток днів у напрямку тренду)
        daily_changes = data['close'].diff().dropna()
        days_in_trend_direction = sum(1 for change in daily_changes if np.sign(change) == trend_direction)
        consistency = days_in_trend_direction / len(daily_changes) if len(daily_changes) > 0 else 0

        # 3. Волатильність (використовуємо стандартне відхилення)
        volatility = data['close'].pct_change().std()
        volatility_normalized = min(1.0, volatility * 20)  # Нормалізуємо до діапазону 0-1

        # 4. Розраховуємо ADX, якщо є необхідні дані
        adx_value = 0.5  # За замовчуванням
        if all(col in data.columns for col in ['high', 'low']):
            # Створюємо "замінник" для ADX на основі простого методу
            high_low_range = (data['high'] - data['low']).mean()
            price_range = data['close'].max() - data['close'].min()
            if price_range > 0:
                adx_value = min(1.0, high_low_range / (price_range * 0.1))

        # Обчислюємо загальну силу тренду (зважена сума різних факторів)
        weights = {
            'r_squared': 0.4,  # Важливість лінійної регресії
            'consistency': 0.3,  # Важливість консистентності
            'volatility': -0.1,  # Негативний вплив надмірної волатильності
            'adx': 0.2  # Важливість ADX або його заміни
        }

        trend_strength = (
                weights['r_squared'] * r_squared +
                weights['consistency'] * consistency +
                weights['volatility'] * (1 - volatility_normalized) +  # Менша волатильність = сильніший тренд
                weights['adx'] * adx_value
        )

        # Нормалізуємо до діапазону від 0 до 1
        trend_strength = max(0, min(1, trend_strength))

        return float(trend_strength)

    def detect_trend_reversal(self, data: pd.DataFrame) -> List[Dict]:

        if data.empty or len(data) < 30:  # Потрібно достатньо історичних даних
            return []

        # Перевіряємо наявність необхідних колонок
        required_columns = ['close', 'high', 'low']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {', '.join(required_columns)}")

        # Підготовка даних
        if 'date' not in data.columns:
            data = data.reset_index()
            if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index

        reversals = []

        # 1. Обчислюємо ковзні середні для визначення тренду
        data['sma20'] = data['close'].rolling(window=20).mean()
        data['sma50'] = data['close'].rolling(window=50).mean()

        # 2. Обчислюємо RSI для визначення перекупленості/перепроданості
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # 3. Знаходимо swing points для визначення локальних максимумів/мінімумів
        swing_points = self.find_swing_points(data, window_size=5)

        # 4. Аналізуємо можливі сигнали розвороту
        for i in range(50, len(data)):
            current_date = data['date'].iloc[i]
            current_price = data['close'].iloc[i]

            # Визначаємо поточний тренд за перетином ковзних середніх
            current_trend = 'uptrend' if data['sma20'].iloc[i] > data['sma50'].iloc[i] else 'downtrend'
            prev_trend = 'uptrend' if data['sma20'].iloc[i - 1] > data['sma50'].iloc[i - 1] else 'downtrend'

            reversal_signal = None
            signal_strength = 0.0

            # Перетин ковзних середніх (сильний сигнал розвороту)
            if current_trend != prev_trend:
                # Перевіряємо чи це перетин зверху вниз (розворот висхідного тренду)
                if current_trend == 'downtrend':
                    reversal_signal = 'bearish_crossover'
                    signal_strength = 0.7
                # Перевіряємо чи це перетин знизу вгору (розворот низхідного тренду)
                else:
                    reversal_signal = 'bullish_crossover'
                    signal_strength = 0.7

            # Перевіряємо на дивергенцію між ціною та RSI
            elif i >= 5:
                # Знаходимо локальні екстремуми ціни та RSI
                price_trend = 'up' if data['close'].iloc[i] > data['close'].iloc[i - 5] else 'down'
                rsi_trend = 'up' if data['rsi'].iloc[i] > data['rsi'].iloc[i - 5] else 'down'

                # Класична дивергенція (різнонаправлений рух ціни та RSI)
                if price_trend != rsi_trend:
                    if price_trend == 'up' and rsi_trend == 'down':
                        reversal_signal = 'bearish_divergence'
                        signal_strength = 0.6
                    else:
                        reversal_signal = 'bullish_divergence'
                        signal_strength = 0.6

            # Додаємо перекупленість/перепроданість по RSI
            if data['rsi'].iloc[i] > 70 and current_trend == 'uptrend':
                # Перевіряємо чи був RSI вище 70 і почав падати
                if data['rsi'].iloc[i] < data['rsi'].iloc[i - 1]:
                    if not reversal_signal:  # Якщо сигналу ще не було
                        reversal_signal = 'overbought'
                        signal_strength = 0.5
                    else:  # Посилюємо існуючий сигнал
                        signal_strength += 0.2

            elif data['rsi'].iloc[i] < 30 and current_trend == 'downtrend':
                # Перевіряємо чи був RSI нижче 30 і почав рости
                if data['rsi'].iloc[i] > data['rsi'].iloc[i - 1]:
                    if not reversal_signal:  # Якщо сигналу ще не було
                        reversal_signal = 'oversold'
                        signal_strength = 0.5
                    else:  # Посилюємо існуючий сигнал
                        signal_strength += 0.2

            # Якщо виявлено сигнал розвороту, додаємо його до списку
            if reversal_signal:
                reversal = {
                    'date': current_date,
                    'price': current_price,
                    'signal_type': reversal_signal,
                    'strength': min(1.0, signal_strength),  # Обмежуємо силу сигналу до 1.0
                    'current_trend': current_trend,
                    'rsi': data['rsi'].iloc[i],
                }

                # Додаємо додаткові дані, якщо доступні
                if 'volume' in data.columns:
                    reversal['volume'] = data['volume'].iloc[i]

                reversals.append(reversal)

        return reversals

    def calculate_fibonacci_levels(self, data: pd.DataFrame,
                                   trend_type: str) -> Dict[str, float]:

        if data.empty or len(data) < 2:
            return {}

        # Перевіряємо наявність необхідних колонок
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {', '.join(required_columns)}")

        # Перевіряємо тип тренду
        if trend_type not in ['uptrend', 'downtrend']:
            raise ValueError("trend_type повинен бути 'uptrend' або 'downtrend'")

        # Знаходимо останні максимуми і мінімуми для тренду
        # Використаємо вікно останніх 90 точок або всі дані, якщо їх менше
        window = min(90, len(data))
        recent_data = data.iloc[-window:]

        # Визначаємо swing high та swing low в межах вікна
        swing_points = self.find_swing_points(recent_data, window_size=5)

        # Якщо не знайдено точок розвороту, використовуємо глобальні максимум і мінімум вікна
        if not swing_points['highs'] and not swing_points['lows']:
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
        else:
            # Для висхідного тренду: беремо останній мінімум і після нього максимум
            # Для низхідного тренду: беремо останній максимум і після нього мінімум
            if trend_type == 'uptrend':
                # Сортуємо точки за індексом (часом)
                lows = sorted(swing_points['lows'], key=lambda x: x['index'])
                highs = sorted(swing_points['highs'], key=lambda x: x['index'])

                # Якщо є точки, знаходимо останній мінімум
                if lows:
                    last_low_index = lows[-1]['index']
                    swing_low = lows[-1]['price']

                    # Знаходимо максимум після останнього мінімуму
                    highs_after_low = [h for h in highs if h['index'] > last_low_index]
                    if highs_after_low:
                        # Беремо найвищий максимум після останнього мінімуму
                        swing_high = max(h['price'] for h in highs_after_low)
                    else:
                        # Якщо немає максимумів після мінімуму, беремо поточну ціну закриття
                        swing_high = data['close'].iloc[-1]
                else:
                    # Якщо немає мінімумів, використовуємо глобальні значення
                    swing_low = recent_data['low'].min()
                    swing_high = recent_data['high'].max()
            else:  # downtrend
                # Сортуємо точки за індексом (часом)
                highs = sorted(swing_points['highs'], key=lambda x: x['index'])
                lows = sorted(swing_points['lows'], key=lambda x: x['index'])

                # Якщо є точки, знаходимо останній максимум
                if highs:
                    last_high_index = highs[-1]['index']
                    swing_high = highs[-1]['price']

                    # Знаходимо мінімум після останнього максимуму
                    lows_after_high = [l for l in lows if l['index'] > last_high_index]
                    if lows_after_high:
                        # Беремо найнижчий мінімум після останнього максимуму
                        swing_low = min(l['price'] for l in lows_after_high)
                    else:
                        # Якщо немає мінімумів після максимуму, беремо поточну ціну закриття
                        swing_low = data['close'].iloc[-1]
                else:
                    # Якщо немає максимумів, використовуємо глобальні значення
                    swing_high = recent_data['high'].max()
                    swing_low = recent_data['low'].min()

        # Переконуємося, що swing_high > swing_low
        if swing_high <= swing_low:
            # У рідкісних випадках може виникнути помилка у визначенні точок
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()

        # Розраховуємо діапазон цін
        price_range = swing_high - swing_low

        # Стандартні рівні Фібоначчі
        fib_levels = {
            '0.0': swing_low,
            '0.236': swing_low + 0.236 * price_range,
            '0.382': swing_low + 0.382 * price_range,
            '0.5': swing_low + 0.5 * price_range,
            '0.618': swing_low + 0.618 * price_range,
            '0.786': swing_low + 0.786 * price_range,
            '1.0': swing_high,
            # Розширені рівні Фібоначчі
            '1.272': swing_low + 1.272 * price_range,
            '1.618': swing_low + 1.618 * price_range,
            '2.618': swing_low + 2.618 * price_range
        }

        # Для низхідного тренду ми можемо "віддзеркалити" рівні
        if trend_type == 'downtrend':
            fib_levels = {
                '0.0': swing_high,
                '0.236': swing_high - 0.236 * price_range,
                '0.382': swing_high - 0.382 * price_range,
                '0.5': swing_high - 0.5 * price_range,
                '0.618': swing_high - 0.618 * price_range,
                '0.786': swing_high - 0.786 * price_range,
                '1.0': swing_low,
                # Розширені рівні Фібоначчі
                '1.272': swing_high - 1.272 * price_range,
                '1.618': swing_high - 1.618 * price_range,
                '2.618': swing_high - 2.618 * price_range
            }

        # Додаємо інформацію про використані екстремуми
        fib_levels['swing_high'] = swing_high
        fib_levels['swing_low'] = swing_low

        return fib_levels

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
        """
        Оцінка тривалості поточного тренду.

        Args:
            data: Цінові дані у форматі DataFrame

        Returns:
            Dict[str, int]: Інформація про тривалість тренду
        """
        pass

    def identify_market_regime(self, data: pd.DataFrame) -> str:
        """
        Визначення поточного режиму ринку
        (тренд, консолідація, висока волатильність).

        Args:
            data: Цінові дані у форматі DataFrame

        Returns:
            str: Режим ринку
        """
        pass

    def detect_divergence(self, price_data: pd.DataFrame,
                          indicator_data: pd.DataFrame) -> List[Dict]:
        """
        Виявлення дивергенцій між ціною та технічними індикаторами.

        Args:
            price_data: Цінові дані у форматі DataFrame
            indicator_data: Дані індикатора (RSI, MACD тощо)

        Returns:
            List[Dict]: Список виявлених дивергенцій
        """
        pass

    def calculate_trend_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Розрахунок метрик поточного тренду (швидкість, прискорення, волатильність).

        Args:
            data: Цінові дані у форматі DataFrame

        Returns:
            Dict[str, float]: Словник метрик тренду
        """
        pass

    def save_trend_analysis_to_db(self, symbol: str, timeframe: str,
                                  analysis_results: Dict) -> bool:
        """
        Збереження результатів аналізу тренду в базу даних.

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал аналізу
            analysis_results: Результати аналізу тренду

        Returns:
            bool: Успішність збереження
        """
        pass

    def load_trend_analysis_from_db(self, symbol: str,
                                    timeframe: str,
                                    start_time: Optional[str] = None,
                                    end_time: Optional[str] = None) -> Dict:
        """
        Завантаження результатів аналізу тренду з бази даних.

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал аналізу
            start_time: Початок періоду
            end_time: Кінець періоду

        Returns:
            Dict: Результати аналізу тренду
        """
        pass

    def get_trend_summary(self, symbol: str, timeframe: str) -> Dict:
        """
        Отримання зведеної інформації про поточний тренд.

        Args:
            symbol: Символ криптовалюти
            timeframe: Часовий інтервал аналізу

        Returns:
            Dict: Зведена інформація про тренд
        """
        pass