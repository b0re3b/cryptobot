import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from featureengineering import (
    DimensionalityReducer,
    FeatureEngineering,
    StatisticalFeatures,
    TechnicalFeatures,
    TimeFeatures
)


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100).cumsum(),
        'high': np.random.normal(105, 5, 100).cumsum(),
        'low': np.random.normal(95, 5, 100).cumsum(),
        'close': np.random.normal(100, 5, 100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100),
    }, index=dates)
    return data


@pytest.fixture
def sample_data_with_nan():
    """Create sample data with NaN values"""
    data = sample_data()
    data.iloc[::10] = np.nan  # Set every 10th row to NaN
    return data


@pytest.fixture
def sample_data_with_target():
    """Create sample data with target variable"""
    data = sample_data()
    data['target'] = np.random.normal(0, 1, 100)
    return data


# DimensionalityReducer Tests
class TestDimensionalityReducer:
    def test_select_features_f_regression(self, sample_data_with_target):
        reducer = DimensionalityReducer()
        X = sample_data_with_target.drop('target', axis=1)
        y = sample_data_with_target['target']

        X_selected, features = reducer.select_features(X, y, n_features=3, method='f_regression')

        assert isinstance(X_selected, pd.DataFrame)
        assert len(features) == 3
        assert all(col in X.columns for col in features)

    def test_select_features_mutual_info(self, sample_data_with_target):
        reducer = DimensionalityReducer()
        X = sample_data_with_target.drop('target', axis=1)
        y = sample_data_with_target['target']

        X_selected, features = reducer.select_features(X, y, n_features=2, method='mutual_info')

        assert isinstance(X_selected, pd.DataFrame)
        assert len(features) == 2

    def test_select_features_rfe(self, sample_data_with_target):
        reducer = DimensionalityReducer()
        X = sample_data_with_target.drop('target', axis=1)
        y = sample_data_with_target['target']

        X_selected, features = reducer.select_features(X, y, n_features=2, method='rfe')

        assert isinstance(X_selected, pd.DataFrame)
        assert len(features) == 2

    def test_reduce_dimensions_pca(self, sample_data):
        reducer = DimensionalityReducer()
        result_df, transformer = reducer.reduce_dimensions(sample_data, n_components=2, method='pca')

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape[1] == 2
        assert hasattr(transformer, 'explained_variance_ratio_')

    def test_reduce_dimensions_kmeans(self, sample_data):
        reducer = DimensionalityReducer()
        result_df, transformer = reducer.reduce_dimensions(sample_data, n_components=3, method='kmeans')

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape[1] == 3
        assert hasattr(transformer, 'cluster_centers_')

    def test_create_polynomial_features(self, sample_data):
        reducer = DimensionalityReducer()
        result_df = reducer.create_polynomial_features(sample_data, degree=2)

        assert isinstance(result_df, pd.DataFrame)
        assert any(col.startswith('poly_') for col in result_df.columns)

    def test_create_cluster_features_kmeans(self, sample_data):
        reducer = DimensionalityReducer()
        result_df = reducer.create_cluster_features(sample_data, n_clusters=3, method='kmeans')

        assert isinstance(result_df, pd.DataFrame)
        assert 'cluster_label' in result_df.columns
        assert any(col.startswith('distance_to_cluster_') for col in result_df.columns)


# FeatureEngineering Tests
class TestFeatureEngineering:
    def test_create_target_variable_return(self, sample_data):
        fe = FeatureEngineering()
        result_df = fe.create_target_variable(sample_data, price_column='close', horizon=1, target_type='return')

        assert f'target_return_1p' in result_df.columns
        assert not result_df[f'target_return_1p'].isna().all()

    def test_create_target_variable_direction(self, sample_data):
        fe = FeatureEngineering()
        result_df = fe.create_target_variable(sample_data, price_column='close', horizon=1, target_type='direction')

        assert f'target_direction_1p' in result_df.columns
        assert set(result_df[f'target_direction_1p'].dropna().unique()).issubset({0, 1})

    def test_prepare_features_pipeline(self, sample_data):
        fe = FeatureEngineering()
        features, target = fe.prepare_features_pipeline(
            sample_data,
            target_column='close',
            horizon=1,
            feature_groups=['lagged', 'rolling', 'returns']
        )

        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(features) == len(target)
        assert any('lag_' in col for col in features.columns)
        assert any('rolling_' in col for col in features.columns)
        assert any('return_' in col for col in features.columns)


# StatisticalFeatures Tests
class TestStatisticalFeatures:
    def test_create_volatility_features(self, sample_data):
        sf = StatisticalFeatures()
        result_df = sf.create_volatility_features(sample_data, price_column='close')

        assert isinstance(result_df, pd.DataFrame)
        assert any('volatility_' in col for col in result_df.columns)
        assert any('realized_volatility_' in col for col in result_df.columns)

    def test_create_return_features(self, sample_data):
        sf = StatisticalFeatures()
        result_df = sf.create_return_features(sample_data, price_column='close')

        assert isinstance(result_df, pd.DataFrame)
        assert any('return_' in col for col in result_df.columns)
        assert any('log_return_' in col for col in result_df.columns)
        assert any('direction_' in col for col in result_df.columns)

    def test_create_volume_features(self, sample_data):
        sf = StatisticalFeatures()
        result_df = sf.create_volume_features(sample_data)

        assert isinstance(result_df, pd.DataFrame)
        assert any('volume_ma_' in col for col in result_df.columns)
        assert any('volume_zscore_' in col for col in result_df.columns)


# TechnicalFeatures Tests
class TestTechnicalFeatures:
    def test_create_technical_features(self, sample_data):
        tf = TechnicalFeatures()
        result_df = tf.create_technical_features(sample_data)

        assert isinstance(result_df, pd.DataFrame)
        assert any('sma_' in col for col in result_df.columns)
        assert any('rsi_' in col for col in result_df.columns)

    def test_create_candle_pattern_features(self, sample_data):
        tf = TechnicalFeatures()
        result_df = tf.create_candle_pattern_features(sample_data)

        assert isinstance(result_df, pd.DataFrame)
        assert 'candle_body' in result_df.columns
        assert any('hammer' in col for col in result_df.columns)
        assert any('engulfing' in col for col in result_df.columns)

    def test_create_custom_indicators(self, sample_data):
        tf = TechnicalFeatures()
        result_df = tf.create_custom_indicators(sample_data)

        assert isinstance(result_df, pd.DataFrame)
        assert any('local_max_' in col for col in result_df.columns)
        assert any('volume_zscore_' in col for col in result_df.columns)


# TimeFeatures Tests
class TestTimeFeatures:
    def test_create_lagged_features(self, sample_data):
        tf = TimeFeatures()
        result_df = tf.create_lagged_features(sample_data, columns=['close', 'volume'])

        assert isinstance(result_df, pd.DataFrame)
        assert any('close_lag_' in col for col in result_df.columns)
        assert any('volume_lag_' in col for col in result_df.columns)

    def test_create_rolling_features(self, sample_data):
        tf = TimeFeatures()
        result_df = tf.create_rolling_features(sample_data, columns=['close'], window_sizes=[5, 10])

        assert isinstance(result_df, pd.DataFrame)
        assert any('close_rolling_5_mean' in col for col in result_df.columns)
        assert any('close_rolling_10_std' in col for col in result_df.columns)

    def test_create_ewm_features(self, sample_data):
        tf = TimeFeatures()
        result_df = tf.create_ewm_features(sample_data, columns=['close'], spans=[5, 10])

        assert isinstance(result_df, pd.DataFrame)
        assert any('close_ewm_5_mean' in col for col in result_df.columns)
        assert any('close_ewm_10_std' in col for col in result_df.columns)

    def test_create_datetime_features(self, sample_data):
        tf = TimeFeatures()
        result_df = tf.create_datetime_features(sample_data)

        assert isinstance(result_df, pd.DataFrame)
        assert 'hour' in result_df.columns
        assert 'day_of_week' in result_df.columns
        assert 'month' in result_df.columns

    def test_create_datetime_features_cyclical(self, sample_data):
        tf = TimeFeatures()
        result_df = tf.create_datetime_features(sample_data, cyclical=True)

        assert isinstance(result_df, pd.DataFrame)
        assert 'hour_sin' in result_df.columns
        assert 'hour_cos' in result_df.columns
        assert 'day_of_week_sin' in result_df.columns


# Edge Cases and Error Handling Tests
class TestErrorHandling:
    def test_select_features_invalid_method(self, sample_data_with_target):
        reducer = DimensionalityReducer()
        X = sample_data_with_target.drop('target', axis=1)
        y = sample_data_with_target['target']

        with pytest.raises(ValueError):
            reducer.select_features(X, y, method='invalid_method')

    def test_reduce_dimensions_invalid_method(self, sample_data):
        reducer = DimensionalityReducer()

        with pytest.raises(ValueError):
            reducer.reduce_dimensions(sample_data, method='invalid_method')

    def test_create_target_invalid_type(self, sample_data):
        fe = FeatureEngineering()

        with pytest.raises(ValueError):
            fe.create_target_variable(sample_data, target_type='invalid_type')

    def test_empty_dataframe(self):
        fe = FeatureEngineering()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            fe.create_target_variable(empty_df)

    def test_missing_columns(self):
        fe = FeatureEngineering()
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        with pytest.raises(ValueError):
            fe.create_target_variable(df, price_column='missing_column')