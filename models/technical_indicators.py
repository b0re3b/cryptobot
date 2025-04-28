import pandas as pd
import numpy as np
import ta
import logging
from typing import List, Dict, Optional, Union, Tuple


class TechnicalIndicators:
    """
    Клас для розрахунку технічних індикаторів криптовалютного ринку.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Ініціалізація класу технічних індикаторів.

        Args:
            log_level: Рівень логування
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def calculate_moving_averages(self, data: pd.DataFrame, column: str = 'close',
                                  windows: List[int] = [9, 20, 50, 200]) -> pd.DataFrame:
        """
        Розрахунок різних ковзних середніх.

        Args:
            data: DataFrame з ціновими даними
            column: Стовпець для розрахунку (зазвичай 'close')
            windows: Список періодів для ковзних середніх

        Returns:
            DataFrame з доданими ковзними середніми
        """
        result = data.copy()

        for window in windows:
            # Просте ковзне середнє (SMA)
            result[f'sma_{window}'] = data[column].rolling(window=window).mean()

            # Експоненціальне ковзне середнє (EMA)
            result[f'ema_{window}'] = data[column].ewm(span=window, adjust=False).mean()

        return result

    def calculate_oscillators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахунок популярних осциляторів.

        Args:
            data: DataFrame з OHLCV даними

        Returns:
            DataFrame з доданими осциляторами
        """
        result = data.copy()

        # RSI - Relative Strength Index
        result['rsi_14'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()

        # MACD - Moving Average Convergence Divergence
        macd = ta.trend.MACD(close=data['close'])
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_diff'] = macd.macd_diff()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close'])
        result['stoch_k'] = stoch.stoch()
        result['stoch_d'] = stoch.stoch_signal()

        return result

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахунок індикаторів обсягу.

        Args:
            data: DataFrame з OHLCV даними

        Returns:
            DataFrame з доданими індикаторами обсягу
        """
        result = data.copy()

        # On-Balance Volume (OBV)
        result['obv'] = ta.volume.OnBalanceVolumeIndicator(close=data['close'],
                                                           volume=data['volume']).on_balance_volume()

        # Volume Weighted Average Price (VWAP)
        # Потрібно реалізувати для однієї сесії, тому розраховуємо для кожного дня
        result['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data[
            'volume'].cumsum()

        return result

    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахунок індикаторів тренду.

        Args:
            data: DataFrame з OHLCV даними

        Returns:
            DataFrame з доданими індикаторами тренду
        """
        result = data.copy()

        # ADX - Average Directional Index
        adx = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'])
        result['adx'] = adx.adx()
        result['adx_pos'] = adx.adx_pos()
        result['adx_neg'] = adx.adx_neg()

        # Parabolic SAR
        result['psar'] = ta.trend.PSARIndicator(high=data['high'], low=data['low'], close=data['close']).psar()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=data['close'])
        result['bb_high'] = bollinger.bollinger_hband()
        result['bb_mid'] = bollinger.bollinger_mavg()
        result['bb_low'] = bollinger.bollinger_lband()

        return result

    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Визначення рівнів підтримки та опору.

        Args:
            data: DataFrame з OHLCV даними
            window: Вікно для пошуку локальних екстремумів

        Returns:
            DataFrame з доданими рівнями підтримки та опору
        """
        # Реалізацію можна доповнити за потреби
        pass

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахунок усіх доступних технічних індикаторів.

        Args:
            data: DataFrame з OHLCV даними

        Returns:
            DataFrame з усіма технічними індикаторами
        """
        result = data.copy()
        result = self.calculate_moving_averages(result)
        result = self.calculate_oscillators(result)
        result = self.calculate_volume_indicators(result)
        result = self.calculate_trend_indicators(result)

        return result

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Генерація торгових сигналів на основі технічних індикаторів.

        Args:
            data: DataFrame з розрахованими технічними індикаторами

        Returns:
            DataFrame з доданими торговими сигналами
        """
        result = data.copy()

        # Сигнали перетину ковзних середніх
        result['sma_9_20_cross'] = np.where(result['sma_9'] > result['sma_20'], 1,
                                            np.where(result['sma_9'] < result['sma_20'], -1, 0))

        # Сигнали RSI
        result['rsi_signal'] = np.where(result['rsi_14'] < 30, 1,  # перекупленість
                                        np.where(result['rsi_14'] > 70, -1, 0))  # перепроданість

        # Сигнали MACD
        result['macd_signal_cross'] = np.where(result['macd'] > result['macd_signal'], 1,
                                               np.where(result['macd'] < result['macd_signal'], -1, 0))

        return result