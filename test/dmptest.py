import logging

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from DMP.DataCleaner import DataCleaner
from DMP.market_data_processor import MarketDataProcessor


@pytest.fixture
def sample_data():
    """Фікстура з тестовими даними OHLCV"""
    index = pd.date_range(start='2023-01-01', periods=100, freq='H')
    data = {
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_data_with_issues():
    """Фікстура з тестовими данами, що містять проблеми"""
    index = pd.date_range(start='2023-01-01', periods=50, freq='H')
    data = {
        'open': np.random.uniform(100, 200, 50),
        'high': np.random.uniform(200, 300, 50),
        'low': np.random.uniform(50, 100, 50),
        'close': np.random.uniform(100, 200, 50),
        'volume': np.random.randint(1000, 10000, 50)
    }

    # Додаємо проблеми
    df = pd.DataFrame(data, index=index)
    df.iloc[10, 1] = 0  # high=0
    df.iloc[15, 2] = 0  # low=0
    df.iloc[20, 3] = -1  # close=-1
    df.iloc[25, 4] = -100  # volume=-100
    df.iloc[30, 1] = df.iloc[30, 2] - 10  # high < low
    df.iloc[35, :] = np.nan  # всі значення NaN
    return df


@pytest.fixture
def data_cleaner():
    """Фікстура для DataCleaner"""
    return DataCleaner()


@pytest.fixture
def market_data_processor():
    """Фікстура для MarketDataProcessor"""
    return MarketDataProcessor(log_level=logging.ERROR)  # Зменшуємо логування для тестів


class TestDataCleaner:
    def test_fix_invalid_high_low(self, data_cleaner, sample_data_with_issues):
        """Тест виправлення некоректних high/low значень"""
        df = sample_data_with_issues.copy()
        result = data_cleaner._fix_invalid_high_low(df)

        # Перевіряємо, що high >= low у всіх рядках
        assert (result['high'] >= result['low']).all()

        # Перевіряємо, що нульові значення high/low виправлені
        assert (result['high'] != 0).all()
        assert (result['low'] != 0).all()

        # Перевіряємо, що оригінальні коректні дані не змінені
        valid_rows = ~df.index.isin([10, 15, 20, 25, 30, 35])
        assert df[valid_rows].equals(result[valid_rows])

    def test_remove_outliers(self, data_cleaner, sample_data):
        """Тест видалення викидів"""
        # Додаємо викиди
        df = sample_data.copy()
        df.iloc[5, 1] = 1000  # outlier в high
        df.iloc[10, 2] = 10  # outlier в low
        df.iloc[15, 4] = 50000  # outlier в volume

        result = data_cleaner._remove_outliers(df)

        # Перевіряємо, що викиди замінені на NaN
        assert pd.isna(result.iloc[5, 1])
        assert pd.isna(result.iloc[10, 2])
        assert pd.isna(result.iloc[15, 4])

        # Перевіряємо, що інші дані не змінені
        valid_rows = ~df.index.isin([5, 10, 15])
        assert df[valid_rows].equals(result[valid_rows])

    def test_fix_invalid_values(self, data_cleaner, sample_data_with_issues):
        """Тест виправлення некоректних значень"""
        df = sample_data_with_issues.copy()
        result = data_cleaner._fix_invalid_values(df, essential_cols=['open', 'high', 'low', 'close', 'volume'])

        # Перевіряємо, що всі ціни додатні
        assert (result[['open', 'high', 'low', 'close']] > 0).all().all()

        # Перевіряємо, що об'єм не від'ємний
        assert (result['volume'] >= 0).all()

        # Перевіряємо, що NaN значення заповнені
        assert not result.iloc[35].isna().any()

    def test_add_crypto_specific_features(self, data_cleaner, sample_data):
        """Тест додавання крипто-специфічних ознак"""
        df = sample_data.copy()
        result = data_cleaner.add_crypto_specific_features(df)

        # Перевіряємо, що додані нові колонки
        expected_new_cols = ['volatility', 'volume_volatility_ratio', 'zero_volume',
                             'flat_price', 'price_change_pct', 'body_size_pct']
        for col in expected_new_cols:
            assert col in result.columns

        # Перевіряємо, що оригінальні дані не змінені
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert df[col].equals(result[col])

    def test_clean_data(self, data_cleaner, sample_data_with_issues):
        """Тест комплексного очищення даних"""
        df = sample_data_with_issues.copy()
        result = data_cleaner.clean_data(df)

        # Перевіряємо основні вимоги до очищених даних
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert (result['high'] >= result['low']).all()
        assert (result[['open', 'high', 'low', 'close']] > 0).all().all()
        assert (result['volume'] >= 0).all()
        assert not result.index.duplicated().any()


class TestMarketDataProcessor:
    def test_align_time_series(self, market_data_processor, sample_data):
        """Тест вирівнювання часових рядів"""
        # Створюємо 3 DataFrame з різними часовими індексами
        df1 = sample_data.copy()
        df2 = sample_data.iloc[10:80].copy()
        df3 = sample_data.iloc[20:90].copy()

        # Змінюємо частоту одного з DataFrame
        df3 = df3.resample('2H').mean()

        aligned = market_data_processor.align_time_series([df1, df2, df3])

        # Перевіряємо результати
        assert len(aligned) == 3
        assert len(aligned[0]) == len(aligned[1]) == len(aligned[2])
        assert aligned[0].index.equals(aligned[1].index)
        assert aligned[1].index.equals(aligned[2].index)

    def test_resample_data(self, market_data_processor, sample_data):
        """Тест ресемплінгу даних"""
        # Ресемплінг з 1H до 4H
        result = market_data_processor.resample_data(sample_data, '4H')

        # Перевіряємо результати
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) < len(sample_data)
        assert 'open' in result.columns
        assert 'close' in result.columns

        # Перевіряємо правильність агрегації
        sample_period = sample_data.loc['2023-01-01':'2023-01-01 04:00:00']
        resampled_first = result.iloc[0]

        assert resampled_first['open'] == sample_period['open'].iloc[0]
        assert resampled_first['high'] == sample_period['high'].max()
        assert resampled_first['low'] == sample_period['low'].min()
        assert resampled_first['close'] == sample_period['close'].iloc[-1]
        assert resampled_first['volume'] == sample_period['volume'].sum()

    def test_process_market_data(self, market_data_processor, sample_data):
        """Тест комплексної обробки даних"""
        # Зберігаємо тестові дані в тимчасову базу (мок)
        symbol = 'TEST'
        timeframe = '1h'

        # Мокуємо метод load_data
        market_data_processor.load_data = lambda *args, **kwargs: sample_data.copy()

        results = market_data_processor.process_market_data(
            symbol=symbol,
            timeframe=timeframe,
            save_results=False
        )

        # Перевіряємо результати
        assert isinstance(results, dict)
        assert 'raw_data' in results
        assert 'processed_data' in results
        assert not results['processed_data'].empty

        # Перевіряємо, що оброблені дані мають додаткові ознаки
        assert 'price_change_pct' in results['processed_data'].columns

    def test_validate_market_data(self, market_data_processor, sample_data):
        """Тест валідації даних"""
        # Валідуємо коректні дані
        is_valid, report = market_data_processor.validate_market_data(sample_data)
        assert is_valid
        assert report['validation_passed']

        # Створюємо некоректні дані
        bad_data = sample_data.copy()
        bad_data.iloc[0, 1] = -1  # high=-1
        bad_data.iloc[1, 2] = 0  # low=0
        bad_data.iloc[2, 1] = bad_data.iloc[2, 2] - 1  # high < low

        is_valid, report = market_data_processor.validate_market_data(bad_data)
        assert not is_valid
        assert not report['validation_passed']
        assert len(report['price_issues']) > 0
        assert report['zero_prices'] > 0

    def test_combine_market_datasets(self, market_data_processor, sample_data):
        """Тест об'єднання наборів даних"""
        # Створюємо 2 набори даних
        df1 = sample_data.copy()
        df2 = sample_data.copy()
        df2.columns = [f'df2_{col}' for col in df2.columns]

        combined = market_data_processor.combine_market_datasets({
            'df1': df1,
            'df2': df2
        })

        # Перевіряємо результати
        assert isinstance(combined, pd.DataFrame)
        assert not combined.empty
        assert len(combined.columns) == len(df1.columns) + len(df2.columns)
        assert all(col.startswith('df1_') or col.startswith('df2_')
                   for col in combined.columns if col != combined.index.name)