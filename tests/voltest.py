import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Додаємо шлях до директорії з основним модулем
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Фікстури для тестових даних
@pytest.fixture
def sample_ohlc_data():
    dates = pd.date_range(start="2023-01-01", periods=100)
    return pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100)
    }, index=dates)


@pytest.fixture
def sample_price_series():
    dates = pd.date_range(start="2023-01-01", periods=100)
    return pd.Series(np.random.uniform(100, 200, 100), index=dates)


@pytest.fixture
def sample_returns():
    dates = pd.date_range(start="2023-01-01", periods=100)
    return pd.Series(np.random.normal(0, 0.02, 100), index=dates)


@pytest.fixture
def volatility_analysis():
    # Створюємо моки для всіх залежностей
    with patch('volatility_analysis.DatabaseManager', autospec=True), \
            patch('volatility_analysis.DataCleaner', autospec=True), \
            patch('volatility_analysis.AnomalyDetector', autospec=True), \
            patch('volatility_analysis.FeatureEngineering', autospec=True), \
            patch('volatility_analysis.TimeSeriesModels', autospec=True):
        # Імпортуємо після мокування
        from volatility_analysis import VolatilityAnalysis

        # Створюємо екземпляр класу
        va = VolatilityAnalysis(use_parallel=False)

        # Мокуємо методи бази даних
        va.db_manager = MagicMock()
        va.db_manager.get_klines.return_value = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [101, 102, 103, 104, 105]
        })

        return va


# Тести базових методів волатильності
def test_calculate_historical_volatility(volatility_analysis, sample_price_series):
    window = 14
    result = volatility_analysis.calculate_historical_volatility(sample_price_series, window=window)

    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_price_series) - 1  # Один елемент втрачається через diff
    assert result.iloc[-1] >= 0  # Волатильність не може бути від'ємною

# Інші тести залишаються без змін...