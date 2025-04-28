import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple, Any, Deque
from collections import deque
import time


class RealtimeTechnicalIndicators:
    """
    Клас для розрахунку технічних індикаторів криптовалютного ринку в реальному часі.
    Оптимізований для швидкого оновлення індикаторів при надходженні нових даних.
    """

    def __init__(self, max_window_size: int = 300, instrument: str = "BTCUSDT",
                 timeframe: str = "1m", log_level: int = logging.INFO) -> None:
        """
        Ініціалізація класу технічних індикаторів реального часу.

        Args:
            max_window_size: Максимальний розмір вікна для зберігання даних (для найдовшого індикатора)
            instrument: Торговий інструмент (пара)
            timeframe: Часовий інтервал
            log_level: Рівень логування
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Метаінформація
        self.instrument = instrument
        self.timeframe = timeframe
        self.max_window_size = max_window_size

        # Ініціалізація буферів і кешів
        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        """Ініціалізація всіх буферів даних та кешів"""
        # Основні буфери даних
        self.prices: Deque[float] = deque(maxlen=self.max_window_size)
        self.high_prices: Deque[float] = deque(maxlen=self.max_window_size)
        self.low_prices: Deque[float] = deque(maxlen=self.max_window_size)
        self.volumes: Deque[float] = deque(maxlen=self.max_window_size)

        # Буфери для проміжних розрахунків (для оптимізації)
        self._sma_cache: Dict[int, Optional[float]] = {}
        self._sma_sums: Dict[int, float] = {}

        self._ema_cache: Dict[int, Optional[float]] = {}
        self._ema_multipliers: Dict[int, float] = {}

        # Кеші для інших індикаторів
        self._rsi_cache: Dict[str, Any] = {
            'gains': deque(maxlen=14),
            'losses': deque(maxlen=14),
            'avg_gain': None,
            'avg_loss': None
        }

        self._stoch_cache: Dict[str, Any] = {
            'stoch_k_values': deque(maxlen=3)
        }

        self._macd_cache: Dict[str, Any] = {
            'ema12': None,
            'ema26': None,
            'signal': deque(maxlen=9)
        }

        self._adx_cache: Dict[str, Any] = {
            'dm_plus': deque(maxlen=14),
            'dm_minus': deque(maxlen=14),
            'tr': deque(maxlen=14),
            'di_plus': None,
            'di_minus': None,
            'adx_values': deque(maxlen=14)
        }

        self._psar_cache: Dict[str, Any] = {
            'trend': None,  # 1 для висхідного, -1 для низхідного
            'extreme_point': None,
            'sar': None,
            'acceleration_factor': 0.02,
            'max_acceleration_factor': 0.2
        }

        self._obv_value: float = 0
        self._vwap_sum: float = 0
        self._volume_sum: float = 0

        # Останні розраховані значення
        self.last_indicators: Dict[str, Optional[float]] = {}

        # Часові мітки
        self.last_update_time: Optional[float] = None
        self.period_start_time: Optional[float] = None  # Початок періоду для VWAP

        # Налаштування вікон для індикаторів
        self.ma_windows: List[int] = [9, 20, 50, 200]
        self._initialize_ma_caches()

    def _initialize_ma_caches(self) -> None:
        """Ініціалізація кешів для ковзних середніх"""
        for window in self.ma_windows:
            self._sma_cache[window] = None
            self._sma_sums[window] = 0.0
            self._ema_cache[window] = None
            self._ema_multipliers[window] = 2.0 / (window + 1)

        # Додаткові вікна для MACD
        if 12 not in self._ema_cache:
            self._ema_cache[12] = None
            self._ema_multipliers[12] = 2.0 / (12 + 1)

        if 26 not in self._ema_cache:
            self._ema_cache[26] = None
            self._ema_multipliers[26] = 2.0 / (26 + 1)

    def reset_buffers(self, instrument: Optional[str] = None,
                      timeframe: Optional[str] = None) -> None:
        """
        Повне скидання всіх буферів і кешів (використовується при зміні інструмента або таймфрейму)

        Args:
            instrument: Новий торговий інструмент (якщо змінився)
            timeframe: Новий часовий інтервал (якщо змінився)
        """
        if instrument:
            self.instrument = instrument

        if timeframe:
            self.timeframe = timeframe

        self.logger.info(f"Скидання буферів для {self.instrument} на {self.timeframe}")

        # Скидання всіх буферів і кешів
        self._initialize_buffers()

    def reset_session(self) -> None:
        """Скидання сесійних індикаторів (наприклад, VWAP) для нового дня/сесії"""
        self.period_start_time = self.last_update_time
        self._vwap_sum = 0
        self._volume_sum = 0
        self.last_indicators['vwap'] = None

        self.logger.info(f"Скидання сесії для {self.instrument} на {self.timeframe}")

    def update(self, price: float, high: Optional[float] = None, low: Optional[float] = None,
               volume: float = 0, timestamp: Optional[float] = None) -> Dict[str, Optional[float]]:
        """
        Оновлення всіх індикаторів на основі нової ціни.

        Args:
            price: Поточна ціна закриття
            high: Максимальна ціна за період (якщо None, використовується price)
            low: Мінімальна ціна за період (якщо None, використовується price)
            volume: Обсяг торгів за період
            timestamp: Часова мітка (якщо None, використовується поточний час)

        Returns:
            Словник з актуальними значеннями всіх індикаторів
        """
        current_time = timestamp if timestamp is not None else time.time()

        # Якщо це перше оновлення або новий період
        if self.last_update_time is None:
            self.period_start_time = current_time
            self._vwap_sum = 0
            self._volume_sum = 0

        # Оновлення буферів даних
        high = high if high is not None else price
        low = low if low is not None else price

        self.prices.append(price)
        self.high_prices.append(high)
        self.low_prices.append(low)
        self.volumes.append(volume)

        # Оновлення часової мітки
        self.last_update_time = current_time

        # Обчислення всіх індикаторів
        self._update_moving_averages(price)
        self._update_oscillators(price, high, low)
        self._update_volume_indicators(price, volume)
        self._update_trend_indicators(price, high, low)

        # Генерація торгових сигналів
        self._generate_signals()

        return self.last_indicators

    def _calculate_typical_price(self, price: float, high: float, low: float) -> float:
        """
        Розрахунок типової ціни (Typical Price)

        Args:
            price: Ціна закриття
            high: Максимальна ціна
            low: Мінімальна ціна

        Returns:
            Типова ціна (середнє між high, low і close)
        """
        return (price + high + low) / 3

    def _update_moving_averages(self, price: float) -> None:
        """
        Оновлення ковзних середніх при появі нової ціни

        Args:
            price: Поточна ціна закриття
        """
        for window in self.ma_windows:
            # Оновлення SMA (оптимізовано для постійного перерахунку)
            if len(self.prices) >= window:
                if self._sma_cache[window] is None:
                    # Перше обчислення
                    self._sma_sums[window] = sum(list(self.prices)[-window:])
                    self._sma_cache[window] = self._sma_sums[window] / window
                else:
                    # Інкрементальне оновлення
                    old_value = self.prices[-window] if len(self.prices) > window else 0
                    self._sma_sums[window] = self._sma_sums[window] - old_value + price
                    self._sma_cache[window] = self._sma_sums[window] / window

            # Оновлення EMA (постійне перерахування)
            if self._ema_cache[window] is None and len(self.prices) >= window:
                # Ініціалізація EMA як SMA
                self._ema_cache[window] = self._sma_cache[window]
            elif self._ema_cache[window] is not None:
                # Формула EMA: (Поточна ціна - Попереднє EMA) * Множник + Попереднє EMA
                multiplier = self._ema_multipliers[window]
                self._ema_cache[window] = (price - self._ema_cache[window]) * multiplier + self._ema_cache[window]

            # Збереження результатів
            self.last_indicators[f'sma_{window}'] = self._sma_cache[window]
            self.last_indicators[f'ema_{window}'] = self._ema_cache[window]

        # Додаткові EMA для MACD (12 і 26)
        for window in [12, 26]:
            if window not in self.ma_windows:
                if self._ema_cache[window] is None and len(self.prices) >= window:
                    # Ініціалізація як SMA
                    sma = sum(list(self.prices)[-window:]) / window
                    self._ema_cache[window] = sma
                elif self._ema_cache[window] is not None:
                    # Оновлення EMA
                    multiplier = self._ema_multipliers[window]
                    self._ema_cache[window] = (price - self._ema_cache[window]) * multiplier + self._ema_cache[window]

    def _update_oscillators(self, price: float, high: float, low: float) -> None:
        """
        Оновлення осциляторів при появі нової ціни

        Args:
            price: Поточна ціна закриття
            high: Максимальна ціна
            low: Мінімальна ціна
        """
        # RSI оновлення
        prices_list = list(self.prices)
        if len(prices_list) > 1:
            change = price - prices_list[-2]
            gain = max(0, change)
            loss = max(0, -change)

            self._rsi_cache['gains'].append(gain)
            self._rsi_cache['losses'].append(loss)

            if len(self._rsi_cache['gains']) == 14:
                if self._rsi_cache['avg_gain'] is None:
                    # Перше обчислення
                    self._rsi_cache['avg_gain'] = sum(self._rsi_cache['gains']) / 14
                    self._rsi_cache['avg_loss'] = sum(self._rsi_cache['losses']) / 14
                else:
                    # Плавне оновлення середніх значень
                    self._rsi_cache['avg_gain'] = (self._rsi_cache['avg_gain'] * 13 + gain) / 14
                    self._rsi_cache['avg_loss'] = (self._rsi_cache['avg_loss'] * 13 + loss) / 14

                # Обчислення RSI
                if self._rsi_cache['avg_loss'] == 0:
                    rsi = 100
                else:
                    rs = self._rsi_cache['avg_gain'] / self._rsi_cache['avg_loss']
                    rsi = 100 - (100 / (1 + rs))

                self.last_indicators['rsi_14'] = rsi

        # MACD оновлення
        if self._ema_cache.get(12) is not None and self._ema_cache.get(26) is not None:
            macd = self._ema_cache[12] - self._ema_cache[26]
            self.last_indicators['macd'] = macd

            # Сигнальна лінія MACD (EMA від MACD з періодом 9)
            self._macd_cache['signal'].append(macd)
            if len(self._macd_cache['signal']) == 9:
                if 'macd_signal' not in self.last_indicators:
                    # Перше обчислення сигнальної лінії
                    self.last_indicators['macd_signal'] = sum(self._macd_cache['signal']) / 9
                else:
                    # Оновлення сигнальної лінії
                    self.last_indicators['macd_signal'] = (macd - self.last_indicators['macd_signal']) * (2 / 10) + \
                                                          self.last_indicators['macd_signal']

                self.last_indicators['macd_diff'] = macd - self.last_indicators['macd_signal']

        # Stochastic Oscillator оновлення
        if len(self.prices) >= 14:
            low_14 = min(list(self.low_prices)[-14:])
            high_14 = max(list(self.high_prices)[-14:])

            if high_14 - low_14 != 0:
                stoch_k = ((price - low_14) / (high_14 - low_14)) * 100
            else:
                stoch_k = 50  # Default if no range

            self.last_indicators['stoch_k'] = stoch_k

            # Stochastic %D (3-періодне SMA від %K)
            self._stoch_cache['stoch_k_values'].append(stoch_k)
            if len(self._stoch_cache['stoch_k_values']) == 3:
                self.last_indicators['stoch_d'] = sum(self._stoch_cache['stoch_k_values']) / 3

    def _update_volume_indicators(self, price: float, volume: float) -> None:
        """
        Оновлення індикаторів обсягу при появі нової ціни

        Args:
            price: Поточна ціна закриття
            volume: Обсяг торгів
        """
        # OBV оновлення
        if len(self.prices) > 1:
            prices_list = list(self.prices)
            if price > prices_list[-2]:
                self._obv_value += volume
            elif price < prices_list[-2]:
                self._obv_value -= volume
            # Якщо ціна не змінилася, OBV не змінюється

            self.last_indicators['obv'] = self._obv_value

        # VWAP оновлення (для поточної сесії/дня)
        if len(self.high_prices) > 0 and len(self.low_prices) > 0:
            typical_price = self._calculate_typical_price(price, self.high_prices[-1], self.low_prices[-1])
            self._vwap_sum += typical_price * volume
            self._volume_sum += volume

            if self._volume_sum > 0:
                self.last_indicators['vwap'] = self._vwap_sum / self._volume_sum
            else:
                self.last_indicators['vwap'] = price  # Якщо немає обсягу, використовуємо поточну ціну

    def _update_trend_indicators(self, price: float, high: float, low: float) -> None:
        """
        Оновлення індикаторів тренду при появі нової ціні

        Args:
            price: Поточна ціна закриття
            high: Максимальна ціна
            low: Мінімальна ціна
        """
        # Bollinger Bands оновлення (використовуємо SMA для центральної лінії)
        if self._sma_cache.get(20) is not None and len(self.prices) >= 20:
            prices_list = list(self.prices)[-20:]
            std_dev = np.std(prices_list)

            self.last_indicators['bb_mid'] = self._sma_cache[20]
            self.last_indicators['bb_high'] = self._sma_cache[20] + (2 * std_dev)
            self.last_indicators['bb_low'] = self._sma_cache[20] - (2 * std_dev)

        # ADX оновлення
        self._calculate_adx(price, high, low)

        # Parabolic SAR оновлення
        self._calculate_parabolic_sar(price, high, low)

    def _calculate_adx(self, price: float, high: float, low: float) -> None:
        """
        Розрахунок ADX (Average Directional Index)

        Args:
            price: Поточна ціна закриття
            high: Максимальна ціна
            low: Мінімальна ціна
        """
        # Для розрахунку ADX потрібно мінімум 2 свічки
        if len(self.prices) < 2 or len(self.high_prices) < 2 or len(self.low_prices) < 2:
            self.last_indicators['adx'] = None
            self.last_indicators['adx_pos'] = None
            self.last_indicators['adx_neg'] = None
            return

        # Розрахунок +DM і -DM
        high_prev = self.high_prices[-2]
        low_prev = self.low_prices[-2]

        # +DM: If current high - previous high > previous low - current low, then +DM = current high - previous high, else +DM = 0
        plus_dm = max(0, high - high_prev) if high - high_prev > low_prev - low else 0

        # -DM: If previous low - current low > current high - previous high, then -DM = previous low - current low, else -DM = 0
        minus_dm = max(0, low_prev - low) if low_prev - low > high - high_prev else 0

        # True Range (TR)
        tr = max(high - low, abs(high - self.prices[-2]), abs(low - self.prices[-2]))

        # Додаємо значення до буферів
        self._adx_cache['dm_plus'].append(plus_dm)
        self._adx_cache['dm_minus'].append(minus_dm)
        self._adx_cache['tr'].append(tr)

        # Якщо у нас недостатньо даних для обчислення ADX, виходимо
        if len(self._adx_cache['tr']) < 14:
            self.last_indicators['adx'] = None
            self.last_indicators['adx_pos'] = None
            self.last_indicators['adx_neg'] = None
            return

        # Розрахунок +DI14 і -DI14
        if self._adx_cache['di_plus'] is None or self._adx_cache['di_minus'] is None:
            # Перше обчислення
            tr_sum = sum(self._adx_cache['tr'])
            plus_dm_sum = sum(self._adx_cache['dm_plus'])
            minus_dm_sum = sum(self._adx_cache['dm_minus'])

            self._adx_cache['di_plus'] = 100 * plus_dm_sum / tr_sum if tr_sum > 0 else 0
            self._adx_cache['di_minus'] = 100 * minus_dm_sum / tr_sum if tr_sum > 0 else 0
        else:
            # Інкрементальне оновлення
            tr_14 = self._adx_cache['tr'][0]  # Найстаріше значення
            plus_dm_14 = self._adx_cache['dm_plus'][0]
            minus_dm_14 = self._adx_cache['dm_minus'][0]

            tr_sum = sum(self._adx_cache['tr'])
            plus_dm_sum = sum(self._adx_cache['dm_plus'])
            minus_dm_sum = sum(self._adx_cache['dm_minus'])

            self._adx_cache['di_plus'] = 100 * plus_dm_sum / tr_sum if tr_sum > 0 else 0
            self._adx_cache['di_minus'] = 100 * minus_dm_sum / tr_sum if tr_sum > 0 else 0

        # Розрахунок DX
        di_sum = abs(self._adx_cache['di_plus'] - self._adx_cache['di_minus'])
        di_diff = self._adx_cache['di_plus'] + self._adx_cache['di_minus']
        dx = 100 * di_sum / di_diff if di_diff > 0 else 0

        # Додавання DX до буфера
        self._adx_cache['adx_values'].append(dx)

        # Розрахунок ADX (14-періодне середнє DX)
        if len(self._adx_cache['adx_values']) == 14:
            adx = sum(self._adx_cache['adx_values']) / 14
            self.last_indicators['adx'] = adx
            self.last_indicators['adx_pos'] = self._adx_cache['di_plus']
            self.last_indicators['adx_neg'] = self._adx_cache['di_minus']

    def _calculate_parabolic_sar(self, price: float, high: float, low: float) -> None:
        """
        Розрахунок індикатора Parabolic SAR

        Args:
            price: Поточна ціна закриття
            high: Максимальна ціна
            low: Мінімальна ціна
        """
        # Для розрахунку PSAR потрібно мінімум 2 свічки
        if len(self.prices) < 2 or len(self.high_prices) < 2 or len(self.low_prices) < 2:
            self.last_indicators['psar'] = None
            return

        # Ініціалізація параметрів при першому обчисленні
        if self._psar_cache['trend'] is None:
            # За замовчуванням починаємо з висхідного тренду
            self._psar_cache['trend'] = 1  # 1 для висхідного, -1 для низхідного
            self._psar_cache['sar'] = self.low_prices[-2]  # Початкове значення - попередній мінімум
            self._psar_cache['extreme_point'] = self.high_prices[-1]  # Екстремальна точка - поточний максимум
            self._psar_cache['acceleration_factor'] = 0.02  # Початковий фактор прискорення
        else:
            # Оновлення SAR
            current_sar = self._psar_cache['sar']
            ep = self._psar_cache['extreme_point']
            af = self._psar_cache['acceleration_factor']

            # Розрахунок нового SAR
            new_sar = current_sar + af * (ep - current_sar)

            # Обмеження SAR для висхідного тренду
            if self._psar_cache['trend'] == 1:  # Висхідний тренд
                # SAR не може бути вище мінімумів двох попередніх свічок
                min_low = min(self.low_prices[-2], self.low_prices[-1])
                new_sar = min(new_sar, min_low)

                # Перевірка зміни тренду
                if new_sar > low:
                    # Тренд змінився на низхідний
                    self._psar_cache['trend'] = -1
                    new_sar = max(self.high_prices[-2], high)  # Новий SAR - максимум
                    self._psar_cache['extreme_point'] = low  # Нова екстремальна точка - мінімум
                    self._psar_cache['acceleration_factor'] = 0.02  # Скидання фактора прискорення
                else:
                    # Тренд залишається висхідним
                    if high > self._psar_cache['extreme_point']:
                        self._psar_cache['extreme_point'] = high  # Оновлення екстремальної точки
                        # Збільшення фактора прискорення
                        self._psar_cache['acceleration_factor'] = min(
                            self._psar_cache['acceleration_factor'] + 0.02,
                            self._psar_cache['max_acceleration_factor']
                        )
            else:  # Низхідний тренд
                # SAR не може бути нижче максимумів двох попередніх свічок
                max_high = max(self.high_prices[-2], self.high_prices[-1])
                new_sar = max(new_sar, max_high)

                # Перевірка зміни тренду
                if new_sar < high:
                    # Тренд змінився на висхідний
                    self._psar_cache['trend'] = 1
                    new_sar = min(self.low_prices[-2], low)  # Новий SAR - мінімум
                    self._psar_cache['extreme_point'] = high  # Нова екстремальна точка - максимум
                    self._psar_cache['acceleration_factor'] = 0.02  # Скидання фактора прискорення
                else:
                    # Тренд залишається низхідним
                    if low < self._psar_cache['extreme_point']:
                        self._psar_cache['extreme_point'] = low  # Оновлення екстремальної точки
                        # Збільшення фактора прискорення
                        self._psar_cache['acceleration_factor'] = min(
                            self._psar_cache['acceleration_factor'] + 0.02,
                            self._psar_cache['max_acceleration_factor']
                        )

            # Збереження нового SAR
            self._psar_cache['sar'] = new_sar

        # Збереження поточного значення SAR
        self.last_indicators['psar'] = self._psar_cache['sar']

    def _generate_signals(self) -> None:
        """Генерація торгових сигналів на основі оновлених індикаторів"""
        # Сигнали перетину ковзних середніх
        if (self.last_indicators.get('sma_9') is not None and
                self.last_indicators.get('sma_20') is not None):
            if self.last_indicators['sma_9'] > self.last_indicators['sma_20']:
                self.last_indicators['sma_9_20_cross'] = 1
            elif self.last_indicators['sma_9'] < self.last_indicators['sma_20']:
                self.last_indicators['sma_9_20_cross'] = -1
            else:
                self.last_indicators['sma_9_20_cross'] = 0

        # Сигнали RSI
        if self.last_indicators.get('rsi_14') is not None:
            if self.last_indicators['rsi_14'] < 30:
                self.last_indicators['rsi_signal'] = 1  # Перекуплений (сигнал на покупку)
            elif self.last_indicators['rsi_14'] > 70:
                self.last_indicators['rsi_signal'] = -1  # Перепроданий (сигнал на продаж)
            else:
                self.last_indicators['rsi_signal'] = 0  # Нейтральний

        # Сигнали MACD
        if (self.last_indicators.get('macd') is not None and
                self.last_indicators.get('macd_signal') is not None):
            if self.last_indicators['macd'] > self.last_indicators['macd_signal']:
                self.last_indicators['macd_signal_cross'] = 1  # Сигнал на покупку
            elif self.last_indicators['macd'] < self.last_indicators['macd_signal']:
                self.last_indicators['macd_signal_cross'] = -1  # Сигнал на продаж
            else:
                self.last_indicators['macd_signal_cross'] = 0  # Нейтральний

                # Сигнали Stochastic
            if (self.last_indicators.get('stoch_k') is not None and
                    self.last_indicators.get('stoch_d') is not None):
                if (self.last_indicators['stoch_k'] < 20 and
                        self.last_indicators['stoch_d'] < 20 and
                        self.last_indicators['stoch_k'] > self.last_indicators['stoch_d']):
                    self.last_indicators['stoch_signal'] = 1  # Сигнал на покупку
                elif (self.last_indicators['stoch_k'] > 80 and
                      self.last_indicators['stoch_d'] > 80 and
                      self.last_indicators['stoch_k'] < self.last_indicators['stoch_d']):
                    self.last_indicators['stoch_signal'] = -1  # Сигнал на продаж
                else:
                    self.last_indicators['stoch_signal'] = 0  # Нейтральний

                # Сигнали Bollinger Bands
            if (self.last_indicators.get('bb_high') is not None and
                    self.last_indicators.get('bb_low') is not None):
                current_price = self.prices[-1] if self.prices else None
                if current_price is not None:
                    if current_price > self.last_indicators['bb_high']:
                        self.last_indicators['bb_signal'] = -1  # Ціна вище верхньої смуги (сигнал на продаж)
                    elif current_price < self.last_indicators['bb_low']:
                        self.last_indicators['bb_signal'] = 1  # Ціна нижче нижньої смуги (сигнал на покупку)
                    else:
                        self.last_indicators['bb_signal'] = 0  # Ціна в межах смуг (нейтральний)

                # Сигнали ADX
            if (self.last_indicators.get('adx') is not None and
                    self.last_indicators.get('adx_pos') is not None and
                    self.last_indicators.get('adx_neg') is not None):
                if (self.last_indicators['adx'] > 25 and
                        self.last_indicators['adx_pos'] > self.last_indicators['adx_neg']):
                    self.last_indicators['adx_signal'] = 1  # Сильний висхідний тренд
                elif (self.last_indicators['adx'] > 25 and
                      self.last_indicators['adx_pos'] < self.last_indicators['adx_neg']):
                    self.last_indicators['adx_signal'] = -1  # Сильний низхідний тренд
                else:
                    self.last_indicators['adx_signal'] = 0  # Слабкий тренд або його відсутність

                # Загальний сигнал (комбінація індикаторів)
            signal_keys = ['sma_9_20_cross', 'rsi_signal', 'macd_signal_cross',
                           'stoch_signal', 'bb_signal', 'adx_signal']

            # Підрахунок позитивних і негативних сигналів
            positive_signals = sum(1 for key in signal_keys
                                   if self.last_indicators.get(key, 0) == 1)
            negative_signals = sum(1 for key in signal_keys
                                   if self.last_indicators.get(key, 0) == -1)

            # Визначення загального сигналу
            total_signals = len([key for key in signal_keys if key in self.last_indicators])
            if total_signals > 0:
                if positive_signals > negative_signals:
                    self.last_indicators['overall_signal'] = 1  # Загальний сигнал на покупку
                elif negative_signals > positive_signals:
                    self.last_indicators['overall_signal'] = -1  # Загальний сигнал на продаж
                else:
                    self.last_indicators['overall_signal'] = 0  # Нейтральний сигнал
            else:
                self.last_indicators['overall_signal'] = 0  # Недостатньо даних для сигналу

        def get_all_indicators(self) -> Dict[str, Optional[float]]:
            """
            Отримання всіх поточних значень індикаторів

            Returns:
                Словник з усіма обчисленими індикаторами
            """
            return self.last_indicators.copy()

        def get_indicator(self, indicator_name: str) -> Optional[float]:
            """
            Отримання значення конкретного індикатора

            Args:
                indicator_name: Назва індикатора

            Returns:
                Значення індикатора або None, якщо індикатор не обчислено
            """
            return self.last_indicators.get(indicator_name)

        def get_signals(self) -> Dict[str, int]:
            """
            Отримання всіх поточних торгових сигналів

            Returns:
                Словник з торговими сигналами (1: покупка, -1: продаж, 0: нейтральний)
            """
            signal_keys = ['sma_9_20_cross', 'rsi_signal', 'macd_signal_cross',
                           'stoch_signal', 'bb_signal', 'adx_signal', 'overall_signal']

            return {key: self.last_indicators.get(key, 0) for key in signal_keys}

        def batch_update(self, ohlcv_data: List[Dict[str, float]]) -> Dict[str, List[Optional[float]]]:
            """
            Пакетне оновлення індикаторів на основі декількох OHLCV свічок

            Args:
                ohlcv_data: Список словників з OHLCV даними
                          (повинні мати ключі 'close', 'high', 'low', 'volume', опціонально 'timestamp')

            Returns:
                Словник з історією значень індикаторів
            """
            # Ініціалізація словника для збереження історії індикаторів
            indicator_history = {key: [] for key in self.last_indicators.keys()}

            # Оновлення для кожної свічки
            for candle in ohlcv_data:
                close = candle['close']
                high = candle.get('high', close)
                low = candle.get('low', close)
                volume = candle.get('volume', 0)
                timestamp = candle.get('timestamp', None)

                # Оновлення індикаторів
                updated_indicators = self.update(close, high, low, volume, timestamp)

                # Збереження значень у історію
                for key, value in updated_indicators.items():
                    indicator_history[key].append(value)

            return indicator_history

        def get_current_state(self) -> Dict[str, Any]:
            """
            Отримання поточного стану об'єкта для серіалізації/збереження

            Returns:
                Словник з поточним станом індикаторів і буферів
            """
            return {
                'prices': list(self.prices),
                'high_prices': list(self.high_prices),
                'low_prices': list(self.low_prices),
                'volumes': list(self.volumes),
                'indicators': self.last_indicators.copy(),
                'timestamp': self.last_update_time,
                'period_start': self.period_start_time,
                'instrument': self.instrument,
                'timeframe': self.timeframe
            }

        def load_state(self, state: Dict[str, Any]) -> None:
            """
            Завантаження збереженого стану об'єкта

            Args:
                state: Словник зі збереженим станом (отриманий з get_current_state)
            """
            # Скидання буферів
            self._initialize_buffers()

            # Завантаження метаданих
            self.instrument = state.get('instrument', self.instrument)
            self.timeframe = state.get('timeframe', self.timeframe)
            self.last_update_time = state.get('timestamp')
            self.period_start_time = state.get('period_start')

            # Завантаження буферів даних
            self.prices.extend(state.get('prices', []))
            self.high_prices.extend(state.get('high_prices', []))
            self.low_prices.extend(state.get('low_prices', []))
            self.volumes.extend(state.get('volumes', []))

            # Ініціалізація індикаторів
            if state.get('prices'):
                self._recalculate_all_indicators()

            # Завантаження значень індикаторів
            self.last_indicators = state.get('indicators', {})

        def _recalculate_all_indicators(self) -> None:
            """
            Перерахунок усіх індикаторів на основі поточних буферів даних
            (використовується після завантаження стану)
            """
            # Тимчасово зберігаємо буфери
            prices = list(self.prices)
            highs = list(self.high_prices)
            lows = list(self.low_prices)
            volumes = list(self.volumes)

            # Скидання буферів і кешів
            self._initialize_buffers()

            # Додаємо дані послідовно для коректного обчислення індикаторів
            for i in range(len(prices)):
                self.update(
                    prices[i],
                    highs[i] if i < len(highs) else prices[i],
                    lows[i] if i < len(lows) else prices[i],
                    volumes[i] if i < len(volumes) else 0
                )

        def get_indicator_description(self, indicator_name: str) -> str:
            """
            Отримання опису конкретного індикатора

            Args:
                indicator_name: Назва індикатора

            Returns:
                Опис індикатора
            """
            descriptions = {
                'sma': 'Simple Moving Average - проста ковзна середня',
                'ema': 'Exponential Moving Average - експоненціальна ковзна середня',
                'rsi': 'Relative Strength Index - індекс відносної сили',
                'macd': 'Moving Average Convergence Divergence - конвергенція/дивергенція ковзних середніх',
                'stoch': 'Stochastic Oscillator - стохастичний осцилятор',
                'bb': 'Bollinger Bands - смуги Боллінджера',
                'vwap': 'Volume Weighted Average Price - середньозважена за обсягом ціна',
                'obv': 'On-Balance Volume - балансовий обсяг',
                'adx': 'Average Directional Index - середній індекс спрямованості',
                'psar': 'Parabolic SAR - параболічна система зупинки та розвороту',
            }

            # Пошук відповідного опису за префіксом
            for prefix, description in descriptions.items():
                if indicator_name.startswith(prefix):
                    return description

            return 'Невідомий індикатор'