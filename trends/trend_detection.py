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
        swing_points = self.support.find_swing_points(data, window_size=3)

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
                    df[k] = v

            # 3. Додаємо інші важливі ознаки
            df['trend_strength'] = self.analyzer.calculate_trend_strength(df)
            market_regime = self.analyzer.identify_market_regime(df)

            # Перетворюємо market_regime на числові значення, якщо це не числа
            if isinstance(market_regime, (pd.Series, np.ndarray)):
                if market_regime.dtype == 'object':
                    # Якщо це категоріальні дані, кодуємо їх як числа
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df['market_regime'] = le.fit_transform(market_regime.fillna('unknown'))
                else:
                    df['market_regime'] = market_regime
            else:
                # Якщо це одне значення, створюємо серію
                df['market_regime'] = market_regime

            # 4. Визначаємо колонки для нормалізації
            feature_cols = ['close', 'volume', 'adx', 'plus_di', 'minus_di', 'rsi',
                            'MACD_12_26_9', 'MACDs_12_26_9', 'trend_strength']

            # Додаємо колонки тренду, якщо вони існують
            trend_cols = ['speed_20', 'volatility_20']
            for col in trend_cols:
                if col in df.columns:
                    feature_cols.append(col)

            # Перевіряємо, чи всі колонки існують
            existing_feature_cols = [col for col in feature_cols if col in df.columns]
            if not existing_feature_cols:
                self.logger.warning("No feature columns found in dataframe")
                return None

            # Залишаємо тільки потрібні колонки та видаляємо NaN
            required_cols = existing_feature_cols + ['market_regime']
            df = df[required_cols].dropna()

            if len(df) < lookback_window + 1:
                self.logger.warning(
                    f"Not enough data after cleaning. Need at least {lookback_window + 1}, got {len(df)}")
                return None

            # 5. Нормалізація даних (важливо для LSTM/GRU)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()

            # Нормалізуємо тільки числові ознаки (не market_regime)
            df_normalized = df.copy()
            df_normalized[existing_feature_cols] = scaler.fit_transform(df[existing_feature_cols].astype(np.float32))

            # 6. Створюємо часові вікна для LSTM/GRU
            X, y = [], []
            for i in range(lookback_window, len(df_normalized)):
                # Беремо вікно ознак
                window_features = df_normalized[existing_feature_cols].iloc[i - lookback_window:i].values
                X.append(window_features)

                # Цільове значення (наступна ціна закриття)
                y.append(df_normalized['close'].iloc[i])

            # 7. Конвертуємо в numpy масиви з правильними типами
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            market_regime_array = df_normalized['market_regime'].iloc[lookback_window:].values.astype(np.int32)

            self.logger.info(
                f"Prepared ML features: X shape {X.shape}, y shape {y.shape}, regime shape {market_regime_array.shape}")

            return X, y, market_regime_array

        except Exception as e:
            self.logger.error(f"Error preparing ML features: {str(e)}")
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