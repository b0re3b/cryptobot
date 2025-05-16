import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple, Any, Deque
from collections import deque
import time


class RealtimeTechnicalIndicators:


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
        """Ініціалізація всіх буферів даних та DataFrame для pandas_ta"""
        # Основні буфери даних для OHLCV
        self.timestamps: Deque[float] = deque(maxlen=self.max_window_size)
        self.open_prices: Deque[float] = deque(maxlen=self.max_window_size)
        self.high_prices: Deque[float] = deque(maxlen=self.max_window_size)
        self.low_prices: Deque[float] = deque(maxlen=self.max_window_size)
        self.close_prices: Deque[float] = deque(maxlen=self.max_window_size)
        self.volumes: Deque[float] = deque(maxlen=self.max_window_size)

        # Порожній DataFrame для даних
        self.ohlcv_df = pd.DataFrame()

        # Буфер для параболічного SAR, який вимагає спеціальної обробки
        self._psar_cache: Dict[str, Any] = {
            'trend': None,  # 1 для висхідного, -1 для низхідного
            'extreme_point': None,
            'sar': None,
            'acceleration_factor': 0.02,
            'max_acceleration_factor': 0.2
        }

        # Для VWAP (Volume Weighted Average Price)
        self._vwap_sum: float = 0
        self._volume_sum: float = 0
        self.period_start_time: Optional[float] = None

        # Останні розраховані значення
        self.last_indicators: Dict[str, Optional[float]] = {}

        # Часові мітки
        self.last_update_time: Optional[float] = None

        # Налаштування вікон для ковзних середніх
        self.ma_windows: List[int] = [9, 20, 50, 200]

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

    def _update_dataframe(self) -> None:
        """Оновлення pandas DataFrame з поточних даних для використання з pandas_ta"""
        # Створюємо DataFrame з накопичених даних
        self.ohlcv_df = pd.DataFrame({
            'timestamp': list(self.timestamps),
            'open': list(self.open_prices),
            'high': list(self.high_prices),
            'low': list(self.low_prices),
            'close': list(self.close_prices),
            'volume': list(self.volumes)
        })

        # Встановлюємо timestamp як індекс для коректної роботи pandas_ta
        if not self.ohlcv_df.empty:
            self.ohlcv_df.set_index('timestamp', inplace=True)

    def update(self, price: float, open_price: Optional[float] = None, high: Optional[float] = None,
               low: Optional[float] = None, volume: float = 0, timestamp: Optional[float] = None) -> Dict[
        str, Optional[float]]:
        """
        Оновлення всіх індикаторів на основі нових OHLCV даних.

        Args:
            price: Поточна ціна закриття
            open_price: Ціна відкриття за період (якщо None, використовується попередня ціна закриття)
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

        # Визначення open_price (якщо не вказано)
        if open_price is None:
            if self.close_prices:
                open_price = self.close_prices[-1]
            else:
                open_price = price

        # Оновлення буферів даних
        high = high if high is not None else price
        low = low if low is not None else price

        self.timestamps.append(current_time)
        self.open_prices.append(open_price)
        self.high_prices.append(high)
        self.low_prices.append(low)
        self.close_prices.append(price)
        self.volumes.append(volume)

        # Оновлення часової мітки
        self.last_update_time = current_time

        # Оновлення DataFrame для pandas_ta
        self._update_dataframe()

        # Обчислення всіх індикаторів
        self._calculate_indicators()

        # Оновлення VWAP (окремо, оскільки враховує сесію)
        self._update_vwap(price, high, low, volume)

        # Генерація торгових сигналів
        self._generate_signals()

        return self.last_indicators

    def _calculate_indicators(self) -> None:
        """Розрахунок всіх технічних індикаторів використовуючи pandas_ta"""
        if self.ohlcv_df.empty:
            return

        # Розрахунок SMA для різних періодів
        for window in self.ma_windows:
            sma = ta.sma(self.ohlcv_df['close'], length=window)
            if not sma.empty:
                self.last_indicators[f'sma_{window}'] = sma.iloc[-1]

        # Розрахунок EMA для різних періодів
        for window in self.ma_windows:
            ema = ta.ema(self.ohlcv_df['close'], length=window)
            if not ema.empty:
                self.last_indicators[f'ema_{window}'] = ema.iloc[-1]

        # RSI
        rsi = ta.rsi(self.ohlcv_df['close'], length=14)
        if not rsi.empty:
            self.last_indicators['rsi_14'] = rsi.iloc[-1]

        # MACD
        macd = ta.macd(self.ohlcv_df['close'], fast=12, slow=26, signal=9)
        if not macd.empty:
            self.last_indicators['macd'] = macd['MACD_12_26_9'].iloc[-1]
            self.last_indicators['macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
            self.last_indicators['macd_diff'] = macd['MACDh_12_26_9'].iloc[-1]

        # Stochastic Oscillator
        stoch = ta.stoch(self.ohlcv_df['high'], self.ohlcv_df['low'], self.ohlcv_df['close'], k=14, d=3, smooth_k=3)
        if not stoch.empty:
            self.last_indicators['stoch_k'] = stoch['STOCHk_14_3_3'].iloc[-1]
            self.last_indicators['stoch_d'] = stoch['STOCHd_14_3_3'].iloc[-1]

        # Bollinger Bands
        bbands = ta.bbands(self.ohlcv_df['close'], length=20, std=2)
        if not bbands.empty:
            self.last_indicators['bb_high'] = bbands['BBU_20_2.0'].iloc[-1]
            self.last_indicators['bb_mid'] = bbands['BBM_20_2.0'].iloc[-1]
            self.last_indicators['bb_low'] = bbands['BBL_20_2.0'].iloc[-1]

        # OBV (On-Balance Volume)
        obv = ta.obv(self.ohlcv_df['close'], self.ohlcv_df['volume'])
        if not obv.empty:
            self.last_indicators['obv'] = obv.iloc[-1]

        # ADX (Average Directional Index)
        adx = ta.adx(self.ohlcv_df['high'], self.ohlcv_df['low'], self.ohlcv_df['close'], length=14)
        if not adx.empty:
            self.last_indicators['adx'] = adx['ADX_14'].iloc[-1]
            self.last_indicators['adx_pos'] = adx['DMP_14'].iloc[-1]
            self.last_indicators['adx_neg'] = adx['DMN_14'].iloc[-1]

        # Parabolic SAR
        psar = ta.psar(self.ohlcv_df['high'], self.ohlcv_df['low'], self.ohlcv_df['close'])
        if not psar.empty:
            self.last_indicators['psar'] = psar['PSARl_0.02_0.2'].iloc[-1]
            # Зберігаємо також тренд (висхідний чи низхідний)
            self._psar_cache['trend'] = 1 if psar['PSARaf_0.02_0.2'].iloc[-1] > 0 else -1
            self._psar_cache['sar'] = psar['PSARl_0.02_0.2'].iloc[-1]

    def _update_vwap(self, price: float, high: float, low: float, volume: float) -> None:
        """
        Оновлення VWAP (для поточної сесії/дня)

        Args:
            price: Ціна закриття
            high: Максимальна ціна
            low: Мінімальна ціна
            volume: Обсяг
        """
        typical_price = (price + high + low) / 3
        self._vwap_sum += typical_price * volume
        self._volume_sum += volume

        if self._volume_sum > 0:
            self.last_indicators['vwap'] = self._vwap_sum / self._volume_sum
        else:
            self.last_indicators['vwap'] = price  # Якщо немає обсягу, використовуємо поточну ціну

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
                self.last_indicators.get('bb_low') is not None and
                self.close_prices):
            current_price = self.close_prices[-1]
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
        Отримання всіх розрахованих індикаторів

        Returns:
            Словник із значеннями всіх індикаторів
        """
        return self.last_indicators.copy()

    def get_indicator(self, indicator_name: str) -> Optional[float]:
        """
        Отримання значення конкретного індикатора

        Args:
            indicator_name: Назва індикатора

        Returns:
            Значення індикатора або None, якщо індикатор не існує
        """
        return self.last_indicators.get(indicator_name)

    def get_signals(self) -> Dict[str, int]:
        """
        Отримання всіх торгових сигналів

        Returns:
            Словник із сигналами (-1 для продажу, 0 для нейтрального, 1 для покупки)
        """
        signal_keys = ['sma_9_20_cross', 'rsi_signal', 'macd_signal_cross',
                       'stoch_signal', 'bb_signal', 'adx_signal', 'overall_signal']

        return {key: self.last_indicators.get(key, 0) for key in signal_keys}

    def batch_update(self, ohlcv_data: List[Dict[str, float]]) -> Dict[str, List[Optional[float]]]:
        """
        Пакетне оновлення індикаторів на основі списку OHLCV даних

        Args:
            ohlcv_data: Список словників з OHLCV даними, кожен словник містить
                      'open', 'high', 'low', 'close', 'volume' та опціонально 'timestamp'

        Returns:
            Словник з історією всіх індикаторів
        """
        # Ефективніший підхід: перетворюємо дані у DataFrame для pandas_ta
        df = pd.DataFrame(ohlcv_data)

        # Перевіряємо наявність необхідних стовпців
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'open':
                    df['open'] = df['close'].shift(1)
                    df['open'].iloc[0] = df['close'].iloc[0]
                elif col in ['high', 'low']:
                    df[col] = df['close']
                elif col == 'volume':
                    df[col] = 0

        # Конвертуємо у DataFrame для pandas_ta
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        # Оновлюємо внутрішній DataFrame
        self.ohlcv_df = df

        # Розраховуємо індикатори для всього DataFrame
        self._batch_calculate_indicators()

        # Оновлюємо буфери даних на основі DataFrame
        self._update_buffers_from_df()

        # Збираємо історію індикаторів
        indicator_history = {}
        for key in self.last_indicators.keys():
            if key in self.ohlcv_df.columns:
                indicator_history[key] = self.ohlcv_df[key].tolist()
            else:
                indicator_history[key] = [None] * len(self.ohlcv_df)

        return indicator_history

    def _batch_calculate_indicators(self) -> None:
        """Розрахунок всіх індикаторів для повного DataFrame"""
        if self.ohlcv_df.empty:
            return

        # Додавання основних технічних індикаторів у DataFrame
        # SMA
        for window in self.ma_windows:
            self.ohlcv_df[f'sma_{window}'] = ta.sma(self.ohlcv_df['close'], length=window)

        # EMA
        for window in self.ma_windows:
            self.ohlcv_df[f'ema_{window}'] = ta.ema(self.ohlcv_df['close'], length=window)

        # RSI
        self.ohlcv_df['rsi_14'] = ta.rsi(self.ohlcv_df['close'], length=14)

        # MACD
        macd = ta.macd(self.ohlcv_df['close'], fast=12, slow=26, signal=9)
        self.ohlcv_df['macd'] = macd['MACD_12_26_9']
        self.ohlcv_df['macd_signal'] = macd['MACDs_12_26_9']
        self.ohlcv_df['macd_diff'] = macd['MACDh_12_26_9']

        # Stochastic
        stoch = ta.stoch(self.ohlcv_df['high'], self.ohlcv_df['low'], self.ohlcv_df['close'], k=14, d=3, smooth_k=3)
        self.ohlcv_df['stoch_k'] = stoch['STOCHk_14_3_3']
        self.ohlcv_df['stoch_d'] = stoch['STOCHd_14_3_3']

        # Bollinger Bands
        bbands = ta.bbands(self.ohlcv_df['close'], length=20, std=2)
        self.ohlcv_df['bb_high'] = bbands['BBU_20_2.0']
        self.ohlcv_df['bb_mid'] = bbands['BBM_20_2.0']
        self.ohlcv_df['bb_low'] = bbands['BBL_20_2.0']

        # OBV
        self.ohlcv_df['obv'] = ta.obv(self.ohlcv_df['close'], self.ohlcv_df['volume'])

        # ADX
        adx = ta.adx(self.ohlcv_df['high'], self.ohlcv_df['low'], self.ohlcv_df['close'], length=14)
        self.ohlcv_df['adx'] = adx['ADX_14']
        self.ohlcv_df['adx_pos'] = adx['DMP_14']
        self.ohlcv_df['adx_neg'] = adx['DMN_14']

        # PSAR
        psar = ta.psar(self.ohlcv_df['high'], self.ohlcv_df['low'], self.ohlcv_df['close'])
        self.ohlcv_df['psar'] = psar['PSARl_0.02_0.2']

        # Оновлюємо останні значення індикаторів
        if not self.ohlcv_df.empty:
            last_row = self.ohlcv_df.iloc[-1]
            for col in self.ohlcv_df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    self.last_indicators[col] = last_row[col]

        # Генеруємо сигнали на основі останніх значень
        self._generate_signals()

    def _update_buffers_from_df(self) -> None:
        """Оновлення внутрішніх буферів на основі оновленого DataFrame"""
        # Очищаємо буфери
        self.timestamps.clear()
        self.open_prices.clear()
        self.high_prices.clear()
        self.low_prices.clear()
        self.close_prices.clear()
        self.volumes.clear()

        # Заповнюємо буфери з DataFrame
        timestamp_values = list(self.ohlcv_df.index) if self.ohlcv_df.index.name == 'timestamp' else [time.time() + i
                                                                                                      for i in range(
                len(self.ohlcv_df))]

        for i, (timestamp, row) in enumerate(zip(timestamp_values, self.ohlcv_df.itertuples())):
            self.timestamps.append(timestamp)
            self.open_prices.append(row.open)
            self.high_prices.append(row.high)
            self.low_prices.append(row.low)
            self.close_prices.append(row.close)
            self.volumes.append(row.volume)

        # Оновлюємо останній час оновлення
        if self.timestamps:
            self.last_update_time = self.timestamps[-1]

    def get_current_state(self) -> Dict[str, Any]:
        """
        Отримання поточного стану для збереження

        Returns:
            Словник із поточним станом класу
        """
        return {
            'timestamps': list(self.timestamps),
            'open_prices': list(self.open_prices),
            'high_prices': list(self.high_prices),
            'low_prices': list(self.low_prices),
            'close_prices': list(self.close_prices),
            'volumes': list(self.volumes),
            'indicators': self.last_indicators.copy(),
            'timestamp': self.last_update_time,
            'period_start': self.period_start_time,
            'instrument': self.instrument,
            'timeframe': self.timeframe
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Завантаження стану з раніше збереженого

        Args:
            state: Словник із збереженим станом
        """
        # Скидання буферів
        self._initialize_buffers()

        # Завантаження метаданих
        self.instrument = state.get('instrument', self.instrument)
        self.timeframe = state.get('timeframe', self.timeframe)
        self.last_update_time = state.get('timestamp')
        self.period_start_time = state.get('period_start')

        # Завантаження буферів даних
        self.timestamps.extend(state.get('timestamps', []))
        self.open_prices.extend(state.get('open_prices', []))
        self.high_prices.extend(state.get('high_prices', []))
        self.low_prices.extend(state.get('low_prices', []))
        self.close_prices.extend(state.get('close_prices', []))
        self.volumes.extend(state.get('volumes', []))

        # Завантаження індикаторів
        if 'indicators' in state:
            self.last_indicators = state['indicators'].copy()

        # Оновлення DataFrame для розрахунків
        self._update_dataframe()

        # Оновлення VWAP параметрів
        if 'vwap_sum' in state:
            self._vwap_sum = state['vwap_sum']
        if 'volume_sum' in state:
            self._volume_sum = state['volume_sum']

        # Оновлення PSAR кешу
        if 'psar_cache' in state:
            self._psar_cache = state['psar_cache'].copy()

        self.logger.info(f"Стан завантажено для {self.instrument} на {self.timeframe}")

    def save_state(self, file_path: str) -> None:
        """
        Збереження поточного стану у файл

        Args:
            file_path: Шлях до файлу для збереження
        """
        state = self.get_current_state()

        # Додаємо додаткові дані, які не включені в get_current_state
        state['vwap_sum'] = self._vwap_sum
        state['volume_sum'] = self._volume_sum
        state['psar_cache'] = self._psar_cache.copy()

        try:
            with open(file_path, 'w') as f:
                import json
                json.dump(state, f)
            self.logger.info(f"Стан збережено у файл {file_path}")
        except Exception as e:
            self.logger.error(f"Помилка при збереженні стану: {str(e)}")

    def load_state_from_file(self, file_path: str) -> bool:
        """
        Завантаження стану з файлу

        Args:
            file_path: Шлях до файлу зі збереженим станом

        Returns:
            True якщо завантаження успішне, False в іншому випадку
        """
        try:
            with open(file_path, 'r') as f:
                import json
                state = json.load(f)
            self.load_state(state)
            return True
        except Exception as e:
            self.logger.error(f"Помилка при завантаженні стану: {str(e)}")
            return False

    def get_signal_strength(self) -> Dict[str, float]:
        """
        Отримання сили сигналів на основі індикаторів

        Returns:
            Словник із значеннями сили сигналів (від -1.0 до 1.0)
        """
        strength = {}

        # RSI сила сигналу
        if 'rsi_14' in self.last_indicators:
            rsi = self.last_indicators['rsi_14']
            if rsi is not None:
                # Нормалізація RSI від -1 до 1 (замість 0-100)
                # < 30 вважається перекупленим (сигнал на покупку)
                # > 70 вважається перепроданим (сигнал на продаж)
                if rsi <= 30:
                    strength['rsi'] = (30 - rsi) / 30  # Максимум 1.0 при RSI=0
                elif rsi >= 70:
                    strength['rsi'] = -1.0 * (rsi - 70) / 30  # Мінімум -1.0 при RSI=100
                else:
                    # Нейтральна зона
                    strength['rsi'] = 0.0

        # MACD сила сигналу
        if 'macd' in self.last_indicators and 'macd_signal' in self.last_indicators:
            macd = self.last_indicators['macd']
            signal = self.last_indicators['macd_signal']
            if macd is not None and signal is not None:
                diff = macd - signal
                # Нормалізація різниці
                if abs(diff) > 0:
                    # Визначення максимального діапазону
                    max_range = 2.0  # Типовий діапазон для MACD
                    normalized_diff = diff / max_range
                    # Обмеження до діапазону [-1, 1]
                    strength['macd'] = max(-1.0, min(1.0, normalized_diff))
                else:
                    strength['macd'] = 0.0

        # Bollinger Bands сила сигналу
        if 'bb_high' in self.last_indicators and 'bb_low' in self.last_indicators and self.close_prices:
            bb_high = self.last_indicators['bb_high']
            bb_low = self.last_indicators['bb_low']
            bb_mid = self.last_indicators['bb_mid']
            price = self.close_prices[-1]

            if bb_high is not None and bb_low is not None and bb_mid is not None:
                # Розраховуємо ширину Bollinger Bands
                bb_width = bb_high - bb_low
                if bb_width > 0:
                    # Відхилення від середньої лінії у відсотках від ширини
                    deviation = (price - bb_mid) / bb_width * 2  # Множимо на 2 для досягнення діапазону [-1, 1]
                    # Обмеження до діапазону [-1, 1]
                    strength['bollinger'] = max(-1.0, min(1.0, deviation))
                else:
                    strength['bollinger'] = 0.0

        # Stochastic Oscillator сила сигналу
        if 'stoch_k' in self.last_indicators and 'stoch_d' in self.last_indicators:
            k = self.last_indicators['stoch_k']
            d = self.last_indicators['stoch_d']
            if k is not None and d is not None:
                # Перетворення діапазону 0-100 на -1 до 1
                # < 20 вважається перекупленим (сигнал на покупку)
                # > 80 вважається перепроданим (сигнал на продаж)
                avg = (k + d) / 2
                if avg <= 20:
                    strength['stochastic'] = (20 - avg) / 20  # Максимум 1.0 при avg=0
                elif avg >= 80:
                    strength['stochastic'] = -1.0 * (avg - 80) / 20  # Мінімум -1.0 при avg=100
                else:
                    # Нейтральна зона
                    strength['stochastic'] = 0.0

        # ADX сила сигналу (сила тренду)
        if 'adx' in self.last_indicators and 'adx_pos' in self.last_indicators and 'adx_neg' in self.last_indicators:
            adx = self.last_indicators['adx']
            pos_di = self.last_indicators['adx_pos']
            neg_di = self.last_indicators['adx_neg']

            if adx is not None and pos_di is not None and neg_di is not None:
                # Сила тренду від 0 до 1 на основі ADX
                trend_strength = min(1.0, adx / 50.0)  # ADX > 25 вважається сильним трендом
                # Напрямок тренду (-1 до 1) на основі DI+ та DI-
                if pos_di > neg_di:
                    trend_direction = (pos_di - neg_di) / (pos_di + neg_di)
                elif neg_di > pos_di:
                    trend_direction = -1.0 * (neg_di - pos_di) / (pos_di + neg_di)
                else:
                    trend_direction = 0.0

                strength['adx'] = trend_direction * trend_strength

        # Загальна сила сигналу (середнє значення всіх сигналів)
        if strength:
            strength['overall'] = sum(strength.values()) / len(strength)

        return strength

    def get_trend_info(self) -> Dict[str, Any]:
        """
        Отримання інформації про поточний тренд

        Returns:
            Словник з інформацією про тренд
        """
        trend_info = {
            'direction': 0,  # -1 низхідний, 0 нейтральний, 1 висхідний
            'strength': 0.0,  # від 0.0 до 1.0
            'duration': 0,  # кількість свічок у поточному тренді
            'reason': 'Недостатньо даних'
        }

        # Визначаємо напрямок тренду за допомогою ковзних середніх
        sma_short = self.last_indicators.get('sma_9')
        sma_medium = self.last_indicators.get('sma_20')
        sma_long = self.last_indicators.get('sma_50')

        if sma_short is not None and sma_medium is not None:
            if sma_short > sma_medium:
                trend_info['direction'] = 1
                trend_info['reason'] = 'SMA 9 вище SMA 20'
            elif sma_short < sma_medium:
                trend_info['direction'] = -1
                trend_info['reason'] = 'SMA 9 нижче SMA 20'

        # Підтвердження тренду за допомогою довшої ковзної
        if sma_long is not None and sma_medium is not None:
            if sma_medium > sma_long and trend_info['direction'] == 1:
                trend_info['reason'] += ' та SMA 20 вище SMA 50'
                trend_info['strength'] += 0.2
            elif sma_medium < sma_long and trend_info['direction'] == -1:
                trend_info['reason'] += ' та SMA 20 нижче SMA 50'
                trend_info['strength'] += 0.2

        # Сила тренду за ADX
        adx = self.last_indicators.get('adx')
        if adx is not None:
            if adx > 25:  # Сильний тренд
                trend_info['strength'] += 0.3 + min(0.5, (adx - 25) / 50)
                trend_info['reason'] += f', ADX={adx:.1f} (сильний тренд)'
            else:
                trend_info['strength'] += adx / 100.0
                trend_info['reason'] += f', ADX={adx:.1f} (слабкий тренд)'

        # Оцінка тривалості тренду
        if self.close_prices and len(self.close_prices) > 2:
            # Перевіряємо, скільки послідовних свічок у тренді
            direction = trend_info['direction']
            duration = 0

            # Рахуємо послідовні свічки, що підтверджують тренд
            for i in range(len(self.close_prices) - 1, 0, -1):
                if direction == 1 and self.close_prices[i] >= self.close_prices[i - 1]:
                    duration += 1
                elif direction == -1 and self.close_prices[i] <= self.close_prices[i - 1]:
                    duration += 1
                else:
                    break

            trend_info['duration'] = duration

        # Нормалізація сили тренду до діапазону [0, 1]
        trend_info['strength'] = min(1.0, trend_info['strength'])

        return trend_info

    def get_price_volatility(self, window: int = 14) -> Optional[float]:
        """
        Розрахунок волатильності ціни за вказаний період

        Args:
            window: Розмір вікна для розрахунку волатильності

        Returns:
            Волатильність у відсотках або None, якщо недостатньо даних
        """
        if len(self.close_prices) < window:
            return None

        # Використовуємо останні N свічок
        prices = list(self.close_prices)[-window:]

        if not prices:
            return None

        # Розрахунок стандартного відхилення
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5

        # Волатильність у відсотках
        volatility = (std_dev / mean) * 100

        return volatility

    def get_support_resistance_levels(self, num_levels: int = 3) -> Dict[str, List[float]]:
        """
        Визначення рівнів підтримки та супротиву

        Args:
            num_levels: Кількість рівнів для пошуку

        Returns:
            Словник із рівнями підтримки та супротиву
        """
        levels = {
            'support': [],
            'resistance': []
        }

        if len(self.close_prices) < 20:
            return levels

        prices = np.array(self.close_prices)
        highs = np.array(self.high_prices)
        lows = np.array(self.low_prices)

        # Поточна ціна
        current_price = self.close_prices[-1]

        # Функція для виявлення локальних мінімумів і максимумів
        def find_local_extremes(data, window_size=5):
            local_max = []
            local_min = []

            for i in range(window_size, len(data) - window_size):
                price_window = data[i - window_size:i + window_size + 1]

                # Локальний максимум
                if data[i] == max(price_window):
                    local_max.append((i, data[i]))

                # Локальний мінімум
                if data[i] == min(price_window):
                    local_min.append((i, data[i]))

            return local_max, local_min

        # Знаходимо локальні екстремуми для максимумів і мінімумів
        local_max_high, _ = find_local_extremes(highs)
        _, local_min_low = find_local_extremes(lows)

        # Відбираємо рівні супротиву (вище поточної ціни)
        resistance_levels = [price for _, price in local_max_high if price > current_price]
        resistance_levels.sort()

        # Відбираємо рівні підтримки (нижче поточної ціни)
        support_levels = [price for _, price in local_min_low if price < current_price]
        support_levels.sort(reverse=True)

        # Видаляємо дублікати і близькі рівні (у межах 0.5%)
        def clean_levels(levels):
            if not levels:
                return []

            cleaned = [levels[0]]
            for level in levels[1:]:
                if abs(level - cleaned[-1]) / cleaned[-1] > 0.005:  # Мінімальна відстань 0.5%
                    cleaned.append(level)
            return cleaned

        resistance_levels = clean_levels(resistance_levels)
        support_levels = clean_levels(support_levels)

        # Обмеження кількості рівнів
        levels['resistance'] = resistance_levels[:num_levels]
        levels['support'] = support_levels[:num_levels]

        # Додаємо рівні на основі технічних індикаторів
        if 'bb_high' in self.last_indicators and self.last_indicators['bb_high'] is not None:
            if self.last_indicators['bb_high'] > current_price:
                levels['resistance'].append(self.last_indicators['bb_high'])

        if 'bb_low' in self.last_indicators and self.last_indicators['bb_low'] is not None:
            if self.last_indicators['bb_low'] < current_price:
                levels['support'].append(self.last_indicators['bb_low'])

        return levels

    def predict_price_movement(self) -> Dict[str, Any]:
        """
        Спроба прогнозу руху ціни на основі аналізу індикаторів

        Returns:
            Словник з прогнозом руху ціни
        """
        prediction = {
            'direction': 0,  # -1 вниз, 0 нейтрально, 1 вверх
            'confidence': 0.0,  # від 0.0 до 1.0
            'target_price': None,  # цільова ціна
            'reasons': []  # обґрунтування
        }

        if not self.close_prices:
            return prediction

        current_price = self.close_prices[-1]

        # Аналіз сигналів від індикаторів
        signals = self.get_signals()
        signal_strength = self.get_signal_strength()

        # Обчислення загального напрямку
        signal_sum = sum(signals.values())
        if signal_sum > 0:
            prediction['direction'] = 1
        elif signal_sum < 0:
            prediction['direction'] = -1

        # Розрахунок впевненості
        if 'overall' in signal_strength:
            prediction['confidence'] = abs(signal_strength['overall'])

        # Розрахунок цільової ціни
        levels = self.get_support_resistance_levels()

        if prediction['direction'] == 1 and levels['resistance']:
            # Цільова ціна - найближчий рівень опору
            prediction['target_price'] = levels['resistance'][0]
            prediction['reasons'].append(f"Ціль: найближчий рівень опору ({prediction['target_price']:.2f})")
        elif prediction['direction'] == -1 and levels['support']:
            # Цільова ціна - найближчий рівень підтримки
            prediction['target_price'] = levels['support'][0]
            prediction['reasons'].append(f"Ціль: найближчий рівень підтримки ({prediction['target_price']:.2f})")

        # Додаємо пояснення на основі індикаторів
        if signals.get('rsi_signal', 0) != 0:
            rsi = self.last_indicators.get('rsi_14')
            if rsi is not None:
                if signals['rsi_signal'] == 1:
                    prediction['reasons'].append(f"RSI: {rsi:.1f} (перекуплений)")
                else:
                    prediction['reasons'].append(f"RSI: {rsi:.1f} (перепроданий)")

        if signals.get('macd_signal_cross', 0) != 0:
            if signals['macd_signal_cross'] == 1:
                prediction['reasons'].append("MACD перетнув сигнальну лінію знизу вверх")
            else:
                prediction['reasons'].append("MACD перетнув сигнальну лінію зверху вниз")

        if signals.get('stoch_signal', 0) != 0:
            if signals['stoch_signal'] == 1:
                prediction['reasons'].append("Стохастик входить в зону перекупленості")
            else:
                prediction['reasons'].append("Стохастик входить в зону перепроданості")

        # Додаємо інформацію про тренд
        trend_info = self.get_trend_info()
        if trend_info['direction'] != 0:
            trend_direction = "висхідний" if trend_info['direction'] == 1 else "низхідний"
            prediction['reasons'].append(f"Поточний тренд: {trend_direction} (сила: {trend_info['strength']:.2f})")

        return prediction

    def to_json(self) -> str:
        """
        Серіалізація поточного стану в JSON рядок

        Returns:
            JSON рядок з даними
        """
        import json

        # Отримуємо поточний стан
        state = self.get_current_state()

        # Додаємо додаткові параметри
        state['vwap_sum'] = self._vwap_sum
        state['volume_sum'] = self._volume_sum
        state['psar_cache'] = self._psar_cache

        # Конвертуємо в JSON
        return json.dumps(state)

    @classmethod
    def from_json(cls, json_str: str) -> 'RealtimeTechnicalIndicators':
        """
        Створення екземпляру класу з JSON рядка

        Args:
            json_str: JSON рядок із серіалізованим станом

        Returns:
            Новий екземпляр RealtimeTechnicalIndicators
        """
        import json

        # Парсимо JSON
        state = json.loads(json_str)

        # Створюємо новий екземпляр
        instrument = state.get('instrument', 'BTCUSDT')
        timeframe = state.get('timeframe', '1m')
        max_window_size = max(300, len(state.get('timestamps', [])))

        instance = cls(max_window_size=max_window_size,
                       instrument=instrument,
                       timeframe=timeframe)

        # Завантажуємо стан
        instance.load_state(state)

        return instance