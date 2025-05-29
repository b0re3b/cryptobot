import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

# Import the classes to test
from crypto_cycles import CryptoCycles
from featureextractor import FeatureExtractor
from MarketPhaseFeatureExtractor import MarketPhaseFeatureExtractor


# Fixtures for test data
@pytest.fixture
def sample_crypto_data():
    """Generate sample cryptocurrency data for testing"""
    date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    prices = np.sin(np.linspace(0, 10, len(date_rng))) * 1000 + 10000  # Simulate price movements
    volumes = np.random.randint(1000, 10000, len(date_rng))

    df = pd.DataFrame({
        'open': prices + np.random.normal(0, 50, len(date_rng)),
        'high': prices + np.random.normal(0, 50, len(date_rng)) + 100,
        'low': prices + np.random.normal(0, 50, len(date_rng)) - 100,
        'close': prices,
        'volume': volumes
    }, index=date_rng)

    return df


@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    mock = MagicMock()
    mock.get_klines.return_value = None
    mock.get_latest_cycle_features.return_value = None
    return mock


@pytest.fixture
def crypto_cycles_instance(mock_db_connection):
    """Create a CryptoCycles instance with mocked DB connection"""
    cc = CryptoCycles()
    cc.db_connection = mock_db_connection
    return cc


@pytest.fixture
def feature_extractor_instance():
    """Create a FeatureExtractor instance"""
    return FeatureExtractor()


@pytest.fixture
def market_phase_extractor_instance():
    """Create a MarketPhaseFeatureExtractor instance"""
    return MarketPhaseFeatureExtractor()


class TestCryptoCycles:
    def test_ensure_float_df(self, crypto_cycles_instance, sample_crypto_data):
        """Test Decimal to float conversion"""
        # Add Decimal values to test data
        test_data = sample_crypto_data.copy()
        test_data['decimal_col'] = [Decimal(str(x)) for x in test_data['close']]

        # Test conversion
        result = crypto_cycles_instance._ensure_float_df(test_data)

        assert 'decimal_col' in result.columns
        assert isinstance(result['decimal_col'].iloc[0], float)
        assert isinstance(result['close'].iloc[0], float)

    def test_load_processed_data(self, crypto_cycles_instance, sample_crypto_data, mock_db_connection):
        """Test loading processed data with caching"""
        # Setup mock to return our sample data
        mock_db_connection.get_klines.return_value = sample_crypto_data

        # First call - should load from DB
        result = crypto_cycles_instance.load_processed_data(
            symbol='BTCUSDT',
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )

        assert result is not None
        assert len(result) == len(sample_crypto_data)
        mock_db_connection.get_klines.assert_called_once()

        # Second call - should use cache
        cached_result = crypto_cycles_instance.load_processed_data(
            symbol='BTCUSDT',
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )

        # Should not call DB again
        mock_db_connection.get_klines.assert_called_once()
        assert result.equals(cached_result)

    def test_compare_current_to_historical_cycles(self, crypto_cycles_instance, sample_crypto_data):
        """Test cycle comparison functionality"""
        # Add some mock cycle data
        test_data = sample_crypto_data.copy()
        test_data['cycle_id'] = np.where(test_data.index < '2020-07-01', 1, 2)

        # Test with bull_bear cycle type
        result = crypto_cycles_instance.compare_current_to_historical_cycles(
            processed_data=test_data,
            symbol='BTCUSDT',
            cycle_type='bull_bear'
        )

        assert 'similarity_scores' in result
        assert 'current_cycle_length' in result
        assert 'most_similar_cycle' in result

        # Test with auto detection for BTC
        result_auto = crypto_cycles_instance.compare_current_to_historical_cycles(
            processed_data=test_data,
            symbol='BTC',
            cycle_type='auto'
        )

        assert result_auto['most_similar_cycle'] is not None

    def test_predict_cycle_turning_points(self, crypto_cycles_instance, sample_crypto_data):
        """Test turning point prediction"""
        # Add required technical indicators
        test_data = sample_crypto_data.copy()
        test_data['rsi'] = np.where(test_data.index < '2020-06-01', 30, 70)  # Simulate oversold/overbought

        result = crypto_cycles_instance.predict_cycle_turning_points(
            processed_data=test_data,
            symbol='BTCUSDT'
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'direction' in result.columns
        assert 'confidence' in result.columns

    def test_update_features_with_new_data(self, crypto_cycles_instance, sample_crypto_data, mock_db_connection):
        """Test feature updating pipeline"""
        # Setup mock to return empty features (simulating first run)
        mock_db_connection.get_latest_cycle_features.return_value = None

        # Add some initial features
        test_data = sample_crypto_data.copy()
        test_data['market_phase'] = 'accumulation'

        result = crypto_cycles_instance.update_features_with_new_data(
            processed_data=test_data,
            symbol='BTCUSDT'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'market_phase' in result.columns
        mock_db_connection.save_cycle_feature.assert_called()


class TestFeatureExtractor:
    def test_create_cyclical_features(self, feature_extractor_instance, sample_crypto_data):
        """Test cyclical feature creation"""
        result = feature_extractor_instance.create_cyclical_features(
            processed_data=sample_crypto_data,
            symbol='BTCUSDT'
        )

        # Check basic cyclical features were added
        assert 'day_of_week_sin' in result.columns
        assert 'month_cos' in result.columns
        assert 'quarter_sin' in result.columns

        # Check market phase features if detected
        if 'market_phase' in result.columns:
            assert 'phase_accumulation' in result.columns

    def test_find_optimal_cycle_length(self, feature_extractor_instance, sample_crypto_data):
        """Test optimal cycle length detection"""
        cycle_length, strength = feature_extractor_instance.find_optimal_cycle_length(
            processed_data=sample_crypto_data
        )

        assert isinstance(cycle_length, int)
        assert isinstance(strength, float)
        assert cycle_length > 0

    def test_calculate_cycle_roi(self, feature_extractor_instance, sample_crypto_data):
        """Test ROI calculation by cycle"""
        # Add some cycle data
        test_data = sample_crypto_data.copy()
        test_data['cycle_id'] = np.where(test_data.index < '2020-07-01', 1, 2)
        test_data['cycle_state'] = np.where(test_data.index < '2020-07-01', 'bull', 'bear')

        result = feature_extractor_instance.calculate_cycle_roi(
            processed_data=test_data,
            symbol='BTCUSDT'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'cycle_1_bull_roi' in result.columns or 'current_cycle_roi' in result.columns

    def test_detect_cycle_anomalies(self, feature_extractor_instance, sample_crypto_data):
        """Test anomaly detection"""
        # Add some technical indicators
        test_data = sample_crypto_data.copy()
        test_data['volatility_14d'] = test_data['close'].pct_change().rolling(14).std()

        result = feature_extractor_instance.detect_cycle_anomalies(
            processed_data=test_data,
            symbol='BTCUSDT'
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'anomaly_type' in result.columns


class TestMarketPhaseFeatureExtractor:
    def test_detect_market_phase(self, market_phase_extractor_instance, sample_crypto_data):
        """Test market phase detection"""
        result = market_phase_extractor_instance.detect_market_phase(
            processed_data=sample_crypto_data
        )

        assert 'market_phase' in result.columns
        assert not result['market_phase'].isnull().any()

    def test_identify_bull_bear_cycles(self, market_phase_extractor_instance, sample_crypto_data):
        """Test bull/bear cycle identification"""
        result = market_phase_extractor_instance.identify_bull_bear_cycles(
            processed_data=sample_crypto_data
        )

        assert 'cycle_state' in result.columns
        assert 'cycle_id' in result.columns
        assert hasattr(result, 'cycles_summary')

    def test_calculate_volatility_by_cycle_phase(self, market_phase_extractor_instance, sample_crypto_data):
        """Test volatility calculation by cycle phase"""
        result = market_phase_extractor_instance.calculate_volatility_by_cycle_phase(
            processed_data=sample_crypto_data,
            symbol='BTCUSDT'
        )

        assert 'volatility' in result.columns
        assert 'cycle_phase' in result.columns
        assert hasattr(result, 'phase_volatility_summary')