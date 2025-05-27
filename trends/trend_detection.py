from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from data.db import DatabaseManager
from trends.supportresistantanalyzer import SupportResistantAnalyzer
from trends.trend_analyzer import TrendAnalyzer
from utils.logger import CryptoLogger


class TrendDetection:

    def __init__(self, config=None):

        self.db_manager = DatabaseManager()
        self.analyzer = TrendAnalyzer()
        self.support = SupportResistantAnalyzer()
        # Ініціалізація конфігурації
        self.config = config or {}

        # Встановлення значень за замовчуванням, якщо не вказано інше
        self.default_window = self.config.get('default_window', 14)
        self.default_threshold = self.config.get('default_threshold', 0.02)
        self.min_points_for_level = self.config.get('min_points_for_level', 3)

        # Ініціалізація логера
        self.logger = CryptoLogger('trend_detection')

    def detect_trend_reversal(self, data: pd.DataFrame, recent_periods: int = 100) -> List[Dict]:
        """
        Виявляє розвороти тренду з фокусом на останні дані
        """
        if data.empty or len(data) < 30:
            return []

        # Перевіряємо наявність необхідних колонок
        required_columns = ['close', 'high', 'low', 'open']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Дані повинні містити колонки: {', '.join(required_columns)}")

        # Підготовка даних - беремо тільки останні періоди для аналізу
        df = data.tail(recent_periods).copy()

        # Додаємо дату якщо її немає
        if 'date' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['date'] = df.index
            else:
                df['date'] = pd.to_datetime(df.index)

        # 1. Обчислюємо ковзні середні для визначення тренду
        df['sma20'] = ta.sma(df['close'], length=20)
        df['sma50'] = ta.sma(df['close'], length=50)

        # 2. Обчислюємо RSI для визначення перекупленості/перепроданості
        df['rsi'] = ta.rsi(df['close'], length=14)

        # 3. Обчислюємо MACD для підтвердження розвороту
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']

        # 4. Обчислюємо Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']

        # Функції для визначення свічкових паттернів
        def identify_engulfing(df, i):
            if i < 1:
                return None
            # Бичаче поглинання
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                    df['close'].iloc[i - 1] < df['open'].iloc[i - 1] and
                    df['close'].iloc[i] > df['open'].iloc[i - 1] and
                    df['open'].iloc[i] < df['close'].iloc[i - 1]):
                return 'bullish'
            # Ведмеже поглинання
            elif (df['close'].iloc[i] < df['open'].iloc[i] and
                  df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and
                  df['close'].iloc[i] < df['open'].iloc[i - 1] and
                  df['open'].iloc[i] > df['close'].iloc[i - 1]):
                return 'bearish'
            return None

        def identify_hammer(df, i):
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            if body_size == 0:
                return False
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])

            if lower_shadow > 2 * body_size and upper_shadow < 0.5 * body_size:
                return df['close'].iloc[i] >= df['open'].iloc[i]
            return False

        def identify_shooting_star(df, i):
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            if body_size == 0:
                return False
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])

            if upper_shadow > 2 * body_size and lower_shadow < 0.5 * body_size:
                return df['close'].iloc[i] <= df['open'].iloc[i]
            return False

        # Ініціалізуємо список для зберігання результатів
        reversals = []

        # Аналізуємо сигнали розвороту тільки для останніх даних
        start_idx = max(50, len(df) - 50)  # Аналізуємо останні 50 свічок

        for i in range(start_idx, len(df)):
            if i < 1:
                continue

            current_date = df['date'].iloc[i]
            current_price = df['close'].iloc[i]

            # Перевіряємо наявність всіх необхідних значень
            if pd.isna(df['sma20'].iloc[i]) or pd.isna(df['sma50'].iloc[i]):
                continue

            # Визначаємо поточний тренд
            current_trend = 'uptrend' if df['sma20'].iloc[i] > df['sma50'].iloc[i] else 'downtrend'
            prev_trend = 'uptrend' if df['sma20'].iloc[i - 1] > df['sma50'].iloc[i - 1] else 'downtrend'

            reversal_signal = None
            signal_strength = 0.0
            signal_reasons = []

            # 1. Перетин ковзних середніх
            if current_trend != prev_trend:
                if current_trend == 'uptrend':
                    reversal_signal = 'bullish_crossover'
                    signal_strength = 0.7
                    signal_reasons.append("SMA20 перетнула SMA50 знизу вгору")
                else:
                    reversal_signal = 'bearish_crossover'
                    signal_strength = 0.7
                    signal_reasons.append("SMA20 перетнула SMA50 зверху вниз")

            # 2. MACD сигнали
            if not pd.isna(df['macd'].iloc[i]) and not pd.isna(df['macd_signal'].iloc[i]):
                macd_cross_bullish = (df['macd'].iloc[i] > df['macd_signal'].iloc[i] and
                                      df['macd'].iloc[i - 1] <= df['macd_signal'].iloc[i - 1])
                macd_cross_bearish = (df['macd'].iloc[i] < df['macd_signal'].iloc[i] and
                                      df['macd'].iloc[i - 1] >= df['macd_signal'].iloc[i - 1])

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

            # 3. RSI сигнали
            if not pd.isna(df['rsi'].iloc[i]):
                if df['rsi'].iloc[i] > 70 and current_trend == 'uptrend':
                    if i > 0 and df['rsi'].iloc[i] < df['rsi'].iloc[i - 1]:
                        if not reversal_signal:
                            reversal_signal = 'overbought'
                            signal_strength = 0.5
                            signal_reasons.append(f"RSI перекуплений ({df['rsi'].iloc[i]:.1f}) і почав падати")
                        else:
                            signal_strength += 0.15
                            signal_reasons.append(f"Підтверджено перекупленим RSI ({df['rsi'].iloc[i]:.1f})")

                elif df['rsi'].iloc[i] < 30 and current_trend == 'downtrend':
                    if i > 0 and df['rsi'].iloc[i] > df['rsi'].iloc[i - 1]:
                        if not reversal_signal:
                            reversal_signal = 'oversold'
                            signal_strength = 0.5
                            signal_reasons.append(f"RSI перепроданий ({df['rsi'].iloc[i]:.1f}) і почав рости")
                        else:
                            signal_strength += 0.15
                            signal_reasons.append(f"Підтверджено перепроданим RSI ({df['rsi'].iloc[i]:.1f})")

            # 4. Свічкові патерни
            engulfing = identify_engulfing(df, i)
            if engulfing == 'bearish' and current_trend == 'uptrend':
                if not reversal_signal:
                    reversal_signal = 'bearish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Ведмежий свічковий патерн: bearish_engulfing")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено ведмежим свічковим патерном")
            elif engulfing == 'bullish' and current_trend == 'downtrend':
                if not reversal_signal:
                    reversal_signal = 'bullish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Бичачий свічковий патерн: bullish_engulfing")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено бичачим свічковим патерном")

            # Молот і падаюча зірка
            if identify_hammer(df, i) and current_trend == 'downtrend':
                if not reversal_signal:
                    reversal_signal = 'bullish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Бичачий свічковий патерн: hammer")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено бичачим свічковим патерном: hammer")

            if identify_shooting_star(df, i) and current_trend == 'uptrend':
                if not reversal_signal:
                    reversal_signal = 'bearish_candlestick'
                    signal_strength = 0.55
                    signal_reasons.append("Ведмежий свічковий патерн: shooting_star")
                else:
                    signal_strength += 0.15
                    signal_reasons.append("Підтверджено ведмежим свічковим патерном: shooting_star")

            # 5. Bollinger Bands
            if (not pd.isna(df['bb_upper'].iloc[i]) and not pd.isna(df['bb_lower'].iloc[i]) and
                    i > 0 and not pd.isna(df['bb_upper'].iloc[i - 1]) and not pd.isna(df['bb_lower'].iloc[i - 1])):

                # Відбиття від верхньої лінії
                if (current_trend == 'uptrend' and
                        df['close'].iloc[i - 1] > df['bb_upper'].iloc[i - 1] and
                        df['close'].iloc[i] < df['bb_upper'].iloc[i]):
                    if not reversal_signal:
                        reversal_signal = 'bearish_bb_rejection'
                        signal_strength = 0.45
                        signal_reasons.append("Відбиття від верхньої лінії Bollinger Bands")
                    else:
                        signal_strength += 0.1
                        signal_reasons.append("Підтверджено відбиттям від верхньої лінії BB")

                # Відбиття від нижньої лінії
                elif (current_trend == 'downtrend' and
                      df['close'].iloc[i - 1] < df['bb_lower'].iloc[i - 1] and
                      df['close'].iloc[i] > df['bb_lower'].iloc[i]):
                    if not reversal_signal:
                        reversal_signal = 'bullish_bb_rejection'
                        signal_strength = 0.45
                        signal_reasons.append("Відбиття від нижньої лінії Bollinger Bands")
                    else:
                        signal_strength += 0.1
                        signal_reasons.append("Підтверджено відбиттям від нижньої лінії BB")

            # Додаємо сигнал розвороту
            if reversal_signal and signal_strength > 0.3:  # Мінімальний поріг
                reversal = {
                    'date': current_date,
                    'price': current_price,
                    'signal_type': reversal_signal,
                    'strength': min(1.0, signal_strength),
                    'current_trend': current_trend,
                    'rsi': df['rsi'].iloc[i] if not pd.isna(df['rsi'].iloc[i]) else None,
                    'reasons': signal_reasons
                }

                # Додаємо volume якщо є
                if 'volume' in df.columns:
                    reversal['volume'] = df['volume'].iloc[i]

                reversals.append(reversal)

        return reversals

    def detect_chart_patterns(self, data: pd.DataFrame, recent_periods: int = 200) -> List[Dict]:
        """
        Виявляє графічні паттерни з фокусом на останні дані
        """
        if data.empty or len(data) < 30:
            return []

        # Перевіряємо наявність необхідних колонок
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Дані повинні містити колонки 'high', 'low', 'close'")

        # Беремо останні дані для аналізу
        df = data.tail(recent_periods).copy()

        # Додаємо дату якщо її немає
        if 'date' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['date'] = df.index
            else:
                df['date'] = pd.to_datetime(df.index)

        # Знаходимо точки розвороту
        try:
            swing_points = self.support.find_swing_points(df, window_size=5)
        except Exception as e:
            self.logger.error(f"Error finding swing points: {e}")
            return []

        patterns = []

        # Отримуємо відсортовані точки
        highs = sorted(swing_points.get('highs', []), key=lambda x: x.get('index', 0))
        lows = sorted(swing_points.get('lows', []), key=lambda x: x.get('index', 0))

        # Фільтруємо тільки останні точки
        recent_threshold = len(df) - min(100, len(df) // 2)
        recent_highs = [h for h in highs if h.get('index', 0) >= recent_threshold]
        recent_lows = [l for l in lows if l.get('index', 0) >= recent_threshold]

        # 1. Подвійне дно (Double Bottom)
        if len(recent_lows) >= 2:
            for i in range(len(recent_lows) - 1):
                low1 = recent_lows[i]
                low2 = recent_lows[i + 1]

                # Перевіряємо схожість рівнів
                price_diff = abs(low1['price'] - low2['price'])
                avg_price = (low1['price'] + low2['price']) / 2
                price_diff_pct = price_diff / avg_price if avg_price > 0 else 1

                if price_diff_pct < 0.05:  # Збільшили толерантність до 5%
                    # Шукаємо піки між мінімумами
                    between_highs = [h for h in recent_highs
                                     if low1['index'] < h['index'] < low2['index']]

                    if between_highs:
                        middle_high = max(between_highs, key=lambda x: x['price'])
                        min_low_price = min(low1['price'], low2['price'])
                        rise = (middle_high['price'] - min_low_price) / min_low_price

                        if rise > 0.02:  # Знизили поріг до 2%
                            pattern = {
                                'type': 'double_bottom',
                                'start_date': low1['date'],
                                'end_date': low2['date'],
                                'bottom1': {'date': low1['date'], 'price': low1['price']},
                                'bottom2': {'date': low2['date'], 'price': low2['price']},
                                'middle_peak': {'date': middle_high['date'], 'price': middle_high['price']},
                                'strength': min(1.0, rise * 20),
                                'recent': True
                            }
                            patterns.append(pattern)

        # 2. Подвійна вершина (Double Top)
        if len(recent_highs) >= 2:
            for i in range(len(recent_highs) - 1):
                high1 = recent_highs[i]
                high2 = recent_highs[i + 1]

                price_diff = abs(high1['price'] - high2['price'])
                avg_price = (high1['price'] + high2['price']) / 2
                price_diff_pct = price_diff / avg_price if avg_price > 0 else 1

                if price_diff_pct < 0.05:
                    between_lows = [l for l in recent_lows
                                    if high1['index'] < l['index'] < high2['index']]

                    if between_lows:
                        middle_low = min(between_lows, key=lambda x: x['price'])
                        max_high_price = max(high1['price'], high2['price'])
                        drop = (max_high_price - middle_low['price']) / middle_low['price']

                        if drop > 0.02:
                            pattern = {
                                'type': 'double_top',
                                'start_date': high1['date'],
                                'end_date': high2['date'],
                                'top1': {'date': high1['date'], 'price': high1['price']},
                                'top2': {'date': high2['date'], 'price': high2['price']},
                                'middle_trough': {'date': middle_low['date'], 'price': middle_low['price']},
                                'strength': min(1.0, drop * 20),
                                'recent': True
                            }
                            patterns.append(pattern)

        # 3. Трикутники - аналізуємо останні точки
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            last_highs = recent_highs[-3:] if len(recent_highs) >= 3 else recent_highs
            last_lows = recent_lows[-3:] if len(recent_lows) >= 3 else recent_lows

            # Висхідний трикутник
            if len(last_highs) >= 2:
                high_prices = [h['price'] for h in last_highs]
                high_range = max(high_prices) - min(high_prices)
                avg_high = sum(high_prices) / len(high_prices)

                if high_range / avg_high < 0.05:  # Горизонтальний опір
                    # Перевіряємо зростання мінімумів
                    if len(last_lows) >= 2:
                        low_trend = all(last_lows[i]['price'] <= last_lows[i + 1]['price']
                                        for i in range(len(last_lows) - 1))
                        if low_trend:
                            start_date = min(last_highs[0]['date'], last_lows[0]['date'])
                            end_date = max(last_highs[-1]['date'], last_lows[-1]['date'])

                            pattern = {
                                'type': 'ascending_triangle',
                                'start_date': start_date,
                                'end_date': end_date,
                                'resistance_level': avg_high,
                                'strength': 0.7,
                                'recent': True
                            }
                            patterns.append(pattern)

            # Низхідний трикутник
            if len(last_lows) >= 2:
                low_prices = [l['price'] for l in last_lows]
                low_range = max(low_prices) - min(low_prices)
                avg_low = sum(low_prices) / len(low_prices)

                if low_range / avg_low < 0.05:  # Горизонтальна підтримка
                    # Перевіряємо спадання максимумів
                    if len(last_highs) >= 2:
                        high_trend = all(last_highs[i]['price'] >= last_highs[i + 1]['price']
                                         for i in range(len(last_highs) - 1))
                        if high_trend:
                            start_date = min(last_highs[0]['date'], last_lows[0]['date'])
                            end_date = max(last_highs[-1]['date'], last_lows[-1]['date'])

                            pattern = {
                                'type': 'descending_triangle',
                                'start_date': start_date,
                                'end_date': end_date,
                                'support_level': avg_low,
                                'strength': 0.7,
                                'recent': True
                            }
                            patterns.append(pattern)

        return patterns

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
        """
        Виправлений метод для отримання зведеного звіту тренду
        """
        try:
            # Завантажуємо останній аналіз тренду з бази даних
            analysis_result = self.load_trend_analysis_from_db(
                symbol=symbol,
                timeframe=timeframe,
                latest_only=True
            )

            if analysis_result['status'] != 'success':
                return {
                    'status': 'no_data',
                    'message': f'No trend analysis available for {symbol} {timeframe}',
                    'symbol': symbol,
                    'timeframe': timeframe
                }

            # Отримуємо дані з результату
            data = analysis_result.get('data', {})

            if not data:
                return {
                    'status': 'no_data',
                    'message': 'Empty data returned from database',
                    'symbol': symbol,
                    'timeframe': timeframe
                }

            # Якщо data - це список, беремо перший елемент
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
            elif isinstance(data, dict):
                latest = data
            else:
                return {
                    'status': 'error',
                    'message': 'Invalid data format from database',
                    'symbol': symbol,
                    'timeframe': timeframe
                }

            # Безпечно отримуємо значення з обробкою помилок
            def safe_get(dictionary, key, default=None):
                try:
                    return dictionary.get(key, default)
                except (AttributeError, TypeError):
                    return default

            # Формуємо зведений звіт
            summary = {
                'status': 'success',
                'symbol': symbol,
                'timeframe': timeframe,
                'trend_type': safe_get(latest, 'trend_type', 'unknown'),
                'trend_strength': safe_get(latest, 'trend_strength', 0.0),
                'market_regime': safe_get(latest, 'market_regime', 'unknown'),
                'support_levels': safe_get(latest, 'support_levels', []),
                'resistance_levels': safe_get(latest, 'resistance_levels', []),
                'patterns_detected': len(safe_get(latest, 'detected_patterns', [])),
                'last_analysis_date': safe_get(latest, 'analysis_date'),
                'additional_metrics': safe_get(latest, 'additional_metrics', {})
            }

            # Додаємо метрики з additional_metrics, якщо вони є
            additional_metrics = summary.get('additional_metrics', {})
            if isinstance(additional_metrics, dict):
                summary.update({
                    'trend_speed': safe_get(additional_metrics, 'speed_20', 0.0),
                    'trend_acceleration': safe_get(additional_metrics, 'acceleration_20', 0.0),
                    'volatility': safe_get(additional_metrics, 'volatility_20', 0.0)
                })

            # Додаємо підсумкову інформацію
            summary['summary_text'] = self._generate_trend_summary_text(summary)

            return summary

        except Exception as e:
            self.logger.error(f"Error generating trend summary: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error generating trend summary: {str(e)}',
                'symbol': symbol,
                'timeframe': timeframe
            }

    def _generate_trend_summary_text(self, summary: Dict) -> str:
        """
        Генерує текстовий опис тренду
        """
        try:
            trend_type = summary.get('trend_type', 'unknown')
            trend_strength = summary.get('trend_strength', 0)
            market_regime = summary.get('market_regime', 'unknown')
            patterns_count = summary.get('patterns_detected', 0)

            # Визначаємо силу тренду
            if trend_strength > 0.7:
                strength_desc = "сильний"
            elif trend_strength > 0.4:
                strength_desc = "помірний"
            else:
                strength_desc = "слабкий"

            # Основний опис тренду
            if trend_type == 'uptrend':
                trend_desc = f"Висхідний {strength_desc} тренд"
            elif trend_type == 'downtrend':
                trend_desc = f"Низхідний {strength_desc} тренд"
            else:
                trend_desc = f"Бічний рух, {strength_desc} тренд"

            # Додаємо інформацію про ринковий режим
            regime_desc = f"Ринковий режим: {market_regime}"

            # Додаємо інформацію про паттерни
            if patterns_count > 0:
                pattern_desc = f"Виявлено {patterns_count} графічних паттернів"
            else:
                pattern_desc = "Графічні паттерни не виявлені"

            return f"{trend_desc}. {regime_desc}. {pattern_desc}."

        except Exception as e:
            return f"Помилка генерації опису: {str(e)}"

    def prepare_ml_trend_features(self, data: pd.DataFrame, lookback_window: int = 30) -> Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        try:
            # Копіюємо дані щоб уникнути попереджень
            df = data.copy()

            # 1. Розраховуємо технічні індикатори
            df = self.analyzer.calculate_adx(df)  # ADX, +DI, -DI
            df['rsi'] = ta.rsi(df['close'])  # RSI
            macd = ta.macd(df['close'])  # MACD
            df = pd.concat([df, macd], axis=1)

            # 2. Додаємо метрики тренду
            metrics = self.analyzer.calculate_trend_metrics(df)
            for k, v in metrics.items():
                if v is not None:
                    # Перевіряємо чи це скаляр чи масив
                    if isinstance(v, (int, float)):
                        # Якщо скаляр, створюємо серію з однаковим значенням
                        df[k] = v
                    elif hasattr(v, '__len__') and len(v) == len(df):
                        # Якщо масив відповідної довжини
                        df[k] = v
                    else:
                        # Пропускаємо невідповідні дані
                        continue

            # 3. Додаємо інші важливі ознаки
            trend_strength = self.analyzer.calculate_trend_strength(df)
            if isinstance(trend_strength, (int, float)):
                df['trend_strength'] = trend_strength
            else:
                df['trend_strength'] = trend_strength

            # 4. Обробляємо market_regime
            market_regime = self.analyzer.identify_market_regime(df)

            # Створюємо функцію для кодування market_regime
            def encode_market_regime(regime_value):
                """Кодує market_regime у числове значення"""
                if isinstance(regime_value, str):
                    # Словник для кодування різних режимів
                    regime_mapping = {
                        'trending': 0,
                        'ranging': 1,
                        'volatile': 2,
                        'uptrend': 3,
                        'downtrend': 4,
                        'sideways': 5,
                        'strong_uptrend': 6,
                        'strong_downtrend': 7,
                        'weak_uptrend': 8,
                        'weak_downtrend': 9,
                        'consolidation': 10,
                        'breakout': 11,
                        'normal_volatility': 12,
                        'high_volatility': 13,
                        'low_volatility': 14,
                        'unknown': 15
                    }

                    # Розбиваємо складені назви на частини
                    regime_parts = regime_value.lower().split('_')

                    # Шукаємо відповідність в словнику
                    for part in regime_parts:
                        if part in regime_mapping:
                            return regime_mapping[part]

                    # Якщо не знайдено точної відповідності, повертаємо код для unknown
                    return regime_mapping.get('unknown', 15)
                elif isinstance(regime_value, (int, float)):
                    return int(regime_value)
                else:
                    return 15  # unknown

            # Застосовуємо кодування
            if isinstance(market_regime, (pd.Series, np.ndarray)):
                df['market_regime'] = pd.Series(market_regime).apply(encode_market_regime)
            else:
                # Якщо це одне значення, створюємо серію
                df['market_regime'] = encode_market_regime(market_regime)

            # 5. Визначаємо колонки для нормалізації
            feature_cols = ['close', 'volume']

            # Додаємо технічні індикатори, якщо вони існують
            technical_cols = ['adx', 'plus_di', 'minus_di', 'rsi', 'trend_strength']
            for col in technical_cols:
                if col in df.columns:
                    feature_cols.append(col)

            # Додаємо MACD колонки, якщо вони існують
            macd_cols = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']
            for col in macd_cols:
                if col in df.columns:
                    feature_cols.append(col)

            # Додаємо колонки тренду, якщо вони існують
            trend_cols = ['speed_20', 'volatility_20', 'acceleration_20']
            for col in trend_cols:
                if col in df.columns:
                    feature_cols.append(col)

            # Перевіряємо, чи всі колонки існують
            existing_feature_cols = [col for col in feature_cols if col in df.columns]
            if not existing_feature_cols:
                self.logger.warning("No feature columns found in dataframe")
                return None

            # Залишаємо тільки потрібні колонки
            required_cols = existing_feature_cols + ['market_regime']
            df = df[required_cols].copy()

            # Перевіряємо типи даних та конвертуємо у float
            for col in existing_feature_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Видаляємо рядки з NaN
            df = df.dropna()

            if len(df) < lookback_window + 1:
                self.logger.warning(
                    f"Not enough data after cleaning. Need at least {lookback_window + 1}, got {len(df)}")
                return None

            # 6. Нормалізація даних (важливо для LSTM/GRU)
            from sklearn.preprocessing import MinMaxScaler

            # Створюємо окремі скалери для різних типів ознак
            feature_scaler = MinMaxScaler()

            # Нормалізуємо тільки числові ознаки (не market_regime)
            df_normalized = df.copy()

            # Перевіряємо, чи є дані для нормалізації
            if len(existing_feature_cols) > 0:
                feature_data = df[existing_feature_cols].astype(np.float32)
                df_normalized[existing_feature_cols] = feature_scaler.fit_transform(feature_data)

            # 7. Створюємо часові вікна для LSTM/GRU
            X, y = [], []
            for i in range(lookback_window, len(df_normalized)):
                # Беремо вікно ознак
                window_features = df_normalized[existing_feature_cols].iloc[i - lookback_window:i].values
                X.append(window_features)

                # Цільове значення (наступна ціна закриття)
                y.append(df_normalized['close'].iloc[i])

            # 8. Конвертуємо в numpy масиви з правильними типами
            if len(X) == 0:
                self.logger.warning("No sequences created")
                return None

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            market_regime_array = df_normalized['market_regime'].iloc[lookback_window:].values.astype(np.int32)

            # Перевіряємо розміри
            if X.shape[0] != y.shape[0] or X.shape[0] != market_regime_array.shape[0]:
                self.logger.error(f"Shape mismatch: X {X.shape}, y {y.shape}, regime {market_regime_array.shape}")
                return None

            self.logger.info(
                f"Prepared ML features: X shape {X.shape}, y shape {y.shape}, regime shape {market_regime_array.shape}")

            # Додаткова інформація про ознаки
            self.logger.info(f"Feature columns used: {existing_feature_cols}")
            self.logger.info(f"Market regime encoding examples: {np.unique(market_regime_array)}")

            return X, y, market_regime_array

        except Exception as e:
            self.logger.error(f"Error preparing ML features: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

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
        trend = trend_detector.analyzer.detect_trend(test_data)
        print("\n1. Detected trend:", trend)

        # Test support/resistance levels
        sr_levels = trend_detector.support.identify_support_resistance(test_data)
        print("\n2. Support levels:", [round(x, 2) for x in sr_levels['support'][-3:]])
        print("   Resistance levels:", [round(x, 2) for x in sr_levels['resistance'][-3:]])

        # Test breakouts
        breakouts = trend_detector.support.detect_breakouts(test_data, sr_levels)
        print("\n3. Detected breakouts:", len(breakouts))
        if breakouts:
            latest_breakout = breakouts[-1]
            print(
                f"   Latest breakout: {latest_breakout['type']} at {latest_breakout['level']:.2f} on {latest_breakout['date']}")

        # Test trend strength
        strength = trend_detector.analyzer.calculate_trend_strength(test_data)
        print("\n4. Trend strength (0-1):", round(strength, 2))

        # Test trend reversal detection
        reversals = trend_detector.detect_trend_reversal(test_data)
        print("\n5. Detected reversals:", len(reversals))
        if reversals:
            latest_reversal = reversals[-1]
            print(f"   Latest reversal: {latest_reversal['signal_type']} on {latest_reversal['date']}")

        # Test Fibonacci levels
        fib_levels = trend_detector.support.calculate_fibonacci_levels(test_data, trend)
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
        regime = trend_detector.analyzer.identify_market_regime(test_data)
        print("\n8. Current market regime:", regime)

        # Test trend metrics
        metrics = trend_detector.analyzer.calculate_trend_metrics(test_data)
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