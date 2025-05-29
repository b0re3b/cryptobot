import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import the classes we want to test
from trends.supportresistantanalyzer import SupportResistantAnalyzer
from trends.trend_analyzer import TrendAnalyzer
from trends.trend_detection import TrendDetection


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing"""
    np.random.seed(42)
    date_rng = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
    close_prices = np.cumsum(np.random.randn(len(date_rng)) * 0.5 + 100)

    data = pd.DataFrame({
        'date': date_rng,
        'open': close_prices - np.random.rand(len(date_rng)),
        'high': close_prices + np.random.rand(len(date_rng)),
        'low': close_prices - np.random.rand(len(date_rng)),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, len(date_rng))
    })
    return data.set_index('date')


@pytest.fixture
def sample_trend_data():
    """Generate data with clear trends for testing"""
    # Uptrend data
    uptrend = pd.DataFrame({
        'close': np.linspace(100, 200, 100),
        'high': np.linspace(101, 201, 100),
        'low': np.linspace(99, 199, 100),
        'volume': np.random.randint(1000, 5000, 100)
    })

    # Downtrend data
    downtrend = pd.DataFrame({
        'close': np.linspace(200, 100, 100),
        'high': np.linspace(201, 101, 100),
        'low': np.linspace(199, 99, 100),
        'volume': np.random.randint(1000, 5000, 100)
    })

    # Sideways data
    sideways = pd.DataFrame({
        'close': np.sin(np.linspace(0, 10, 100)) * 10 + 150,
        'high': np.sin(np.linspace(0, 10, 100)) * 10 + 151,
        'low': np.sin(np.linspace(0, 10, 100)) * 10 + 149,
        'volume': np.random.randint(1000, 5000, 100)
    })

    return {
        'uptrend': uptrend,
        'downtrend': downtrend,
        'sideways': sideways
    }


@pytest.fixture
def support_resistance_analyzer():
    return SupportResistantAnalyzer()


@pytest.fixture
def trend_analyzer():
    return TrendAnalyzer()


@pytest.fixture
def trend_detection():
    # Mock the database manager since we don't want to test DB interactions
    with patch('trends.trend_detection.DatabaseManager') as mock_db:
        mock_db.return_value.get_klines.return_value = []
        mock_db.return_value.save_trend_analysis.return_value = True
        mock_db.return_value.get_trend_analysis.return_value = {}
        yield TrendDetection()


class TestSupportResistantAnalyzer:
    def test_group_price_levels(self, support_resistance_analyzer):
        # Test with empty input
        assert support_resistance_analyzer._group_price_levels([], 0.01) == []

        # Test with single point
        assert support_resistance_analyzer._group_price_levels([100], 0.01) == []

        # Test with multiple points that should group
        points = [100, 101, 100.5, 102, 100.8, 100.3, 105, 106, 105.5]
        grouped = support_resistance_analyzer._group_price_levels(points, 0.02)
        assert len(grouped) == 2  # Should have two groups

        # Test with custom min_points_for_level
        analyzer = SupportResistantAnalyzer({'min_points_for_level': 2})
        grouped = analyzer._group_price_levels(points, 0.02)
        assert len(grouped) == 3  # Now should have three groups

    def test_identify_support_resistance(self, support_resistance_analyzer, sample_ohlc_data):
        # Test with empty data
        assert support_resistance_analyzer.identify_support_resistance(pd.DataFrame()) == {"support": [],
                                                                                           "resistance": []}

        # Test with valid data
        result = support_resistance_analyzer.identify_support_resistance(sample_ohlc_data)
        assert isinstance(result, dict)
        assert 'support' in result
        assert 'resistance' in result
        assert isinstance(result['support'], list)
        assert isinstance(result['resistance'], list)

        # Test with only close prices
        data = sample_ohlc_data[['close']].copy()
        result = support_resistance_analyzer.identify_support_resistance(data)
        assert result != {"support": [], "resistance": []}

    def test_detect_breakouts(self, support_resistance_analyzer, sample_ohlc_data):
        # First identify support/resistance levels
        sr_levels = support_resistance_analyzer.identify_support_resistance(sample_ohlc_data)

        # Test with empty data
        assert support_resistance_analyzer.detect_breakouts(pd.DataFrame(), sr_levels) == []

        # Test with valid data
        breakouts = support_resistance_analyzer.detect_breakouts(sample_ohlc_data, sr_levels)
        assert isinstance(breakouts, list)

        # Test with custom threshold
        breakouts = support_resistance_analyzer.detect_breakouts(sample_ohlc_data, sr_levels, threshold=0.005)
        assert isinstance(breakouts, list)

    def test_calculate_fibonacci_levels(self, support_resistance_analyzer, sample_ohlc_data):
        # Test with empty data
        assert support_resistance_analyzer.calculate_fibonacci_levels(pd.DataFrame(), 'uptrend') == {}

        # Test with uptrend
        result = support_resistance_analyzer.calculate_fibonacci_levels(sample_ohlc_data, 'uptrend')
        assert isinstance(result, dict)
        assert any('fib_' in key for key in result.keys())

        # Test with downtrend
        result = support_resistance_analyzer.calculate_fibonacci_levels(sample_ohlc_data, 'downtrend')
        assert isinstance(result, dict)
        assert any('fib_' in key for key in result.keys())

        # Test with invalid trend type
        with pytest.raises(ValueError):
            support_resistance_analyzer.calculate_fibonacci_levels(sample_ohlc_data, 'invalid')


class TestTrendAnalyzer:
    def test_detect_trend(self, trend_analyzer, sample_trend_data):
        # Test with empty data
        assert trend_analyzer.detect_trend(pd.DataFrame()) == "unknown"

        # Test uptrend
        assert trend_analyzer.detect_trend(sample_trend_data['uptrend']) == "uptrend"

        # Test downtrend
        assert trend_analyzer.detect_trend(sample_trend_data['downtrend']) == "downtrend"

        # Test sideways (should be detected as unknown or sideways)
        result = trend_analyzer.detect_trend(sample_trend_data['sideways'])
        assert result in ["sideways", "unknown"]

    def test_calculate_trend_strength(self, trend_analyzer, sample_trend_data):
        # Test with empty data
        assert trend_analyzer.calculate_trend_strength(pd.DataFrame()) == 0.0

        # Test uptrend
        strength = trend_analyzer.calculate_trend_strength(sample_trend_data['uptrend'])
        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # Should be strong for clear uptrend

        # Test downtrend
        strength = trend_analyzer.calculate_trend_strength(sample_trend_data['downtrend'])
        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # Should be strong for clear downtrend

        # Test sideways
        strength = trend_analyzer.calculate_trend_strength(sample_trend_data['sideways'])
        assert 0.0 <= strength <= 1.0
        assert strength < 0.5  # Should be weak for sideways

    def test_identify_market_regime(self, trend_analyzer, sample_trend_data):
        # Test with empty data
        assert trend_analyzer.identify_market_regime(pd.DataFrame()) == "analysis_error"

        # Test uptrend
        regime = trend_analyzer.identify_market_regime(sample_trend_data['uptrend'])
        assert isinstance(regime, str)
        assert "uptrend" in regime or "trend" in regime

        # Test downtrend
        regime = trend_analyzer.identify_market_regime(sample_trend_data['downtrend'])
        assert isinstance(regime, str)
        assert "downtrend" in regime or "trend" in regime

        # Test sideways
        regime = trend_analyzer.identify_market_regime(sample_trend_data['sideways'])
        assert isinstance(regime, str)
        assert "sideways" in regime or "range" in regime or "consolidation" in regime

    def test_calculate_trend_metrics(self, trend_analyzer, sample_trend_data):
        # Test with empty data
        with pytest.raises(ValueError):
            trend_analyzer.calculate_trend_metrics(pd.DataFrame())

        # Test with minimal data
        minimal_data = pd.DataFrame({'close': [100, 101, 102]})
        metrics = trend_analyzer.calculate_trend_metrics(minimal_data)
        assert isinstance(metrics, dict)
        assert 'speed_5' in metrics

        # Test uptrend
        metrics = trend_analyzer.calculate_trend_metrics(sample_trend_data['uptrend'])
        assert metrics['speed_20'] > 0
        assert metrics['acceleration_20'] >= 0
        assert metrics['trend_strength'] > 0.5

        # Test downtrend
        metrics = trend_analyzer.calculate_trend_metrics(sample_trend_data['downtrend'])
        assert metrics['speed_20'] < 0
        assert metrics['trend_strength'] > 0.5

        # Test sideways
        metrics = trend_analyzer.calculate_trend_metrics(sample_trend_data['sideways'])
        assert abs(metrics['speed_20']) < 0.5
        assert metrics['trend_strength'] < 0.5


class TestTrendDetection:
    def test_detect_trend_reversal(self, trend_detection, sample_ohlc_data):
        # Test with empty data
        assert trend_detection.detect_trend_reversal(pd.DataFrame()) == []

        # Test with valid data
        reversals = trend_detection.detect_trend_reversal(sample_ohlc_data)
        assert isinstance(reversals, list)

        # Test with custom recent_periods
        reversals = trend_detection.detect_trend_reversal(sample_ohlc_data, recent_periods=50)
        assert isinstance(reversals, list)

    def test_detect_chart_patterns(self, trend_detection, sample_ohlc_data):
        # Test with empty data
        assert trend_detection.detect_chart_patterns(pd.DataFrame()) == []

        # Test with valid data
        patterns = trend_detection.detect_chart_patterns(sample_ohlc_data)
        assert isinstance(patterns, list)

        # Test with custom recent_periods
        patterns = trend_detection.detect_chart_patterns(sample_ohlc_data, recent_periods=50)
        assert isinstance(patterns, list)

    def test_detect_divergence(self, trend_detection, sample_ohlc_data):
        # Create indicator data (e.g., RSI)
        indicator_data = pd.DataFrame({
            'rsi': np.sin(np.linspace(0, 10, len(sample_ohlc_data))) * 30 + 50
        }, index=sample_ohlc_data.index)

        # Test with empty data
        assert trend_detection.detect_divergence(pd.DataFrame(), pd.DataFrame()) == []

        # Test with valid data
        divergences = trend_detection.detect_divergence(sample_ohlc_data, indicator_data)
        assert isinstance(divergences, list)

    def test_get_trend_summary(self, trend_detection):
        # Test with mock data
        with patch.object(trend_detection, 'load_trend_analysis_from_db') as mock_load:
            mock_load.return_value = {
                'status': 'success',
                'data': {
                    'trend_type': 'uptrend',
                    'trend_strength': 0.8,
                    'market_regime': 'strong_uptrend',
                    'support_levels': [100, 105],
                    'resistance_levels': [110, 115],
                    'detected_patterns': ['double_bottom'],
                    'analysis_date': datetime.now().isoformat(),
                    'additional_metrics': {
                        'speed_20': 0.5,
                        'acceleration_20': 0.1,
                        'volatility_20': 0.2
                    }
                }
            }

            summary = trend_detection.get_trend_summary('BTC', '1d')
            assert summary['status'] == 'success'
            assert summary['trend_type'] == 'uptrend'
            assert summary['trend_strength'] == 0.8
            assert 'summary_text' in summary

    def test_prepare_ml_trend_features(self, trend_detection, sample_ohlc_data):
        # Test with empty data
        assert trend_detection.prepare_ml_trend_features(pd.DataFrame()) is None

        # Test with valid data
        result = trend_detection.prepare_ml_trend_features(sample_ohlc_data)
        assert result is not None
        assert len(result) == 3
        X, y, regimes = result
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(regimes, np.ndarray)
        assert len(X) == len(y) == len(regimes)

        # Test with custom lookback_window
        result = trend_detection.prepare_ml_trend_features(sample_ohlc_data, lookback_window=10)
        assert result is not None
        assert result[0].shape[1] == 10  # Should have 10 time steps