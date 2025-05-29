import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from analysis.volatility_analysis import VolatilityAnalysis


@pytest.fixture
def volatility_analyzer():
    return VolatilityAnalysis(use_parallel=False)


@pytest.fixture
def sample_ohlc_data():
    dates = pd.date_range('2023-01-01', periods=100)
    data = {
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(100, 120, 100),
        'low': np.random.uniform(80, 100, 100),
        'close': np.random.uniform(90, 110, 100),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns():
    return pd.Series(np.random.normal(0, 0.02, 100))


def test_calc_log_returns(volatility_analyzer):
    prices = [100, 105, 110, 115, 120]
    log_returns = volatility_analyzer._calc_log_returns(prices)
    expected = np.log(np.array([105 / 100, 110 / 105, 115 / 110, 120 / 115]))
    assert np.allclose(log_returns, expected)


def test_safe_convert_to_float(volatility_analyzer):
    # Test with pandas Series
    series = pd.Series(['1.0', '2.0', '3.0'])
    result = volatility_analyzer._safe_convert_to_float(series)
    assert result.dtype == 'float64'

    # Test with numpy array
    arr = np.array(['1.0', '2.0', '3.0'])
    result = volatility_analyzer._safe_convert_to_float(arr)
    assert result.dtype == 'float64'

    # Test with list
    lst = ['1.0', '2.0', '3.0']
    result = volatility_analyzer._safe_convert_to_float(lst)
    assert isinstance(result, np.ndarray)
    assert result.dtype == 'float64'


def test_calculate_historical_volatility(volatility_analyzer, sample_ohlc_data):
    vol = volatility_analyzer.calculate_historical_volatility(sample_ohlc_data['close'], window=14)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(sample_ohlc_data) - 1  # One less due to diff
    assert not vol.isna().all()


def test_calculate_parkinson_volatility(volatility_analyzer, sample_ohlc_data):
    vol = volatility_analyzer.calculate_parkinson_volatility(sample_ohlc_data, window=14)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(sample_ohlc_data)
    assert not vol.isna().all()


def test_calculate_garman_klass_volatility(volatility_analyzer, sample_ohlc_data):
    vol = volatility_analyzer.calculate_garman_klass_volatility(sample_ohlc_data, window=14)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(sample_ohlc_data)
    assert not vol.isna().all()


def test_calculate_yang_zhang_volatility(volatility_analyzer, sample_ohlc_data):
    vol = volatility_analyzer.calculate_yang_zhang_volatility(sample_ohlc_data, window=14)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(sample_ohlc_data)
    assert not vol.isna().all()


def test_detect_volatility_regimes(volatility_analyzer, sample_ohlc_data):
    vol = volatility_analyzer.calculate_historical_volatility(sample_ohlc_data['close'], window=14)
    regimes = volatility_analyzer.detect_volatility_regimes(vol, n_regimes=3)
    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(vol)
    assert regimes.nunique() <= 3  # Should have up to 3 unique values


def test_analyze_volatility_clustering(volatility_analyzer, sample_returns):
    acf_df = volatility_analyzer.analyze_volatility_clustering(sample_returns, max_lag=30)
    assert isinstance(acf_df, pd.DataFrame)
    assert 'lag' in acf_df.columns
    assert 'autocorrelation' in acf_df.columns
    assert len(acf_df) == 31  # lags 0 to 30


def test_calculate_volatility_risk_metrics(volatility_analyzer, sample_returns):
    vol = pd.Series(np.random.uniform(0.01, 0.05, len(sample_returns)))
    metrics = volatility_analyzer.calculate_volatility_risk_metrics(sample_returns, vol)
    assert isinstance(metrics, dict)
    expected_keys = ['var_95', 'var_99', 'cvar_95', 'cvar_99',
                     'vol_of_vol_mean', 'sharpe_ratio', 'max_drawdown']
    assert all(key in metrics for key in expected_keys)


def test_compare_volatility_metrics(volatility_analyzer, sample_ohlc_data):
    result = volatility_analyzer.compare_volatility_metrics(sample_ohlc_data, windows=[7, 14])
    assert isinstance(result, pd.DataFrame)
    expected_cols = ['historical_7d', 'parkinson_7d', 'gk_7d', 'yz_7d',
                     'historical_14d', 'parkinson_14d', 'gk_14d', 'yz_14d']
    assert all(col in result.columns for col in expected_cols)


def test_identify_volatility_breakouts(volatility_analyzer):
    vol_series = pd.Series(np.random.uniform(0.01, 0.05, 100))
    breakouts = volatility_analyzer.identify_volatility_breakouts(vol_series, window=20)
    assert isinstance(breakouts, pd.Series)
    assert len(breakouts) == len(vol_series)
    assert breakouts.dtype == bool


def test_extract_seasonality_in_volatility(volatility_analyzer):
    dates = pd.date_range('2023-01-01', periods=365)
    vol_series = pd.Series(np.random.uniform(0.01, 0.05, 365), index=dates)

    # Test day of week seasonality
    dow = volatility_analyzer.extract_seasonality_in_volatility(vol_series, period=7)
    assert isinstance(dow, pd.Series)
    assert len(dow) == 7

    # Test month seasonality
    monthly = volatility_analyzer.extract_seasonality_in_volatility(vol_series, period=12)
    assert isinstance(monthly, pd.Series)
    assert len(monthly) == 12


def test_prepare_volatility_features_for_ml(volatility_analyzer, sample_ohlc_data):
    features = volatility_analyzer.prepare_volatility_features_for_ml(sample_ohlc_data)
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(sample_ohlc_data)
    assert 'returns' in features.columns
    assert any('hist_vol_' in col for col in features.columns)


@patch('volatility_analysis.DatabaseManager')
def test_run_full_volatility_analysis(mock_db, volatility_analyzer):
    # Setup mock database
    mock_db_instance = mock_db.return_value
    mock_db_instance.get_klines.return_value = pd.DataFrame({
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(100, 120, 100),
        'low': np.random.uniform(80, 100, 100),
        'close': np.random.uniform(90, 110, 100),
    })

    result = volatility_analyzer.run_full_volatility_analysis('BTC', '1d', save_to_db=False)

    assert isinstance(result, dict)
    assert 'symbol' in result
    assert 'timeframe' in result
    assert 'volatility_data' in result
    assert 'latest_volatility' in result
    assert 'current_regime' in result


def test_get_market_phases(volatility_analyzer):
    # Create test data with 3 assets
    dates = pd.date_range('2023-01-01', periods=100)
    data = {
        'asset1': np.random.uniform(0.01, 0.05, 100),
        'asset2': np.random.uniform(0.01, 0.05, 100),
        'asset3': np.random.uniform(0.01, 0.05, 100),
    }
    volatility_data = pd.DataFrame(data, index=dates)

    phases = volatility_analyzer.get_market_phases(volatility_data, n_regimes=3)

    assert isinstance(phases, pd.Series)
    assert len(phases) == len(volatility_data)
    assert phases.isin(['Low Vol', 'Normal Vol', 'High Vol']).all()


def test_analyze_cross_asset_volatility(volatility_analyzer):
    asset_dict = {
        'BTC': pd.Series(np.random.uniform(10000, 50000, 100)),
        'ETH': pd.Series(np.random.uniform(1000, 3000, 100)),
        'SOL': pd.Series(np.random.uniform(10, 100, 100)),
    }

    corr_matrix = volatility_analyzer.analyze_cross_asset_volatility(asset_dict)

    assert isinstance(corr_matrix, pd.DataFrame)
    assert set(corr_matrix.index) == set(asset_dict.keys())
    assert set(corr_matrix.columns) == set(asset_dict.keys())
    assert (-1 <= corr_matrix.values).all() and (corr_matrix.values <= 1).all()


def test_volatility_impulse_response(volatility_analyzer, sample_returns):
    result = volatility_analyzer.volatility_impulse_response(sample_returns)

    assert isinstance(result, dict)
    assert 'impulse' in result
    assert 'half_life' in result
    assert 'shock_size' in result
    assert 'max_effect' in result
    assert 'decay_rate' in result


@patch('volatility_analysis.DatabaseManager')
def test_analyze_crypto_market_conditions(mock_db, volatility_analyzer):
    # Setup mock database
    mock_db_instance = mock_db.return_value
    mock_db_instance.get_klines.return_value = pd.DataFrame({
        'close': np.random.uniform(10000, 50000, 100)
    })

    result = volatility_analyzer.analyze_crypto_market_conditions(['BTC', 'ETH'])

    assert isinstance(result, dict)
    assert 'average_market_vol' in result
    assert 'vol_trend_30d' in result
    assert 'vol_dispersion' in result
    assert 'vol_correlation' in result
    assert 'market_phase' in result
    assert 'regime_shifts' in result