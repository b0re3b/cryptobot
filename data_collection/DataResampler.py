import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List, Optional, Union
import statsmodels.api as sm


class EnhancedDataResampler:
    def __init__(self, logger, db_manager):
        self.logger = logger
        self.db_manager = db_manager
        self.scalers = {}  # Dictionary to store scalers for different assets and timeframes

    def resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Ресемплінг даних до вказаного інтервалу часу."""
        # (existing method from your code)
        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для ресемплінгу")
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Дані повинні мати DatetimeIndex для ресемплінгу")

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.warning(f"Відсутні необхідні колонки: {missing_cols}")
            return data

        pandas_interval = self.convert_interval_to_pandas_format(target_interval)
        self.logger.info(f"Ресемплінг даних до інтервалу: {target_interval} (pandas формат: {pandas_interval})")

        if len(data) > 1:
            current_interval = pd.Timedelta(data.index[1] - data.index[0])
            estimated_target_interval = self.parse_interval(target_interval)

            if estimated_target_interval < current_interval:
                self.logger.warning(f"Цільовий інтервал ({target_interval}) менший за поточний інтервал даних. "
                                    f"Даунсемплінг неможливий без додаткових даних.")
                return data

        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }

        if 'volume' in data.columns:
            agg_dict['volume'] = 'sum'

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:
                if any(x in col.lower() for x in ['count', 'number', 'trades']):
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'mean'

        try:
            resampled = data.resample(pandas_interval).agg(agg_dict)

            if resampled.isna().any().any():
                self.logger.info("Заповнення відсутніх значень після ресемплінгу...")
                price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in resampled.columns]
                resampled[price_cols] = resampled[price_cols].fillna(method='ffill')

                if 'volume' in resampled.columns:
                    resampled['volume'] = resampled['volume'].fillna(0)

            self.logger.info(f"Ресемплінг успішно завершено: {resampled.shape[0]} рядків")
            return resampled

        except Exception as e:
            self.logger.error(f"Помилка при ресемплінгу даних: {str(e)}")
            raise

    # (Other utility methods from your code)
    def convert_interval_to_pandas_format(self, timeframe: str) -> str:
        # Implementation as in your original code
        interval_map = {
            's': 'S',
            'm': 'T',
            'h': 'H',
            'd': 'D',
            'w': 'W',
            'M': 'M',
        }

        if not timeframe or not isinstance(timeframe, str):
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        if unit in interval_map:
            return f"{number}{interval_map[unit]}"
        else:
            raise ValueError(f"Непідтримувана одиниця часу: {unit}")

    def parse_interval(self, timeframe: str) -> pd.Timedelta:
        # Implementation as in your original code
        interval_map = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
        }

        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        if unit == 'M':
            return pd.Timedelta(days=int(number) * 30.44)

        return pd.Timedelta(**{interval_map[unit]: int(number)})

    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Створює додаткові часові ознаки для моделей машинного навчання."""
        # (existing method from your code with enhancements)
        df = data.copy()

        # Часові компоненти
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Додаємо is_weekend флаг
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday

        # Додаємо торгову сесію
        df['session'] = 'Unknown'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'Asia'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'Europe'
        df.loc[(df['hour'] >= 16) & (df['hour'] < 24), 'session'] = 'US'

        # Циклічні ознаки для часових компонентів
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
        df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
        df['day_of_month_sin'] = np.sin((df['day_of_month'] - 1) * (2 * np.pi / 31))
        df['day_of_month_cos'] = np.cos((df['day_of_month'] - 1) * (2 * np.pi / 31))

        return df

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок технічних індикаторів для аналізу ринку."""
        df = data.copy()

        # Moving Averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std_dev = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
        df['bb_lower'] = df['bb_middle'] - (std_dev * 2)

        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100

        # Trend (positive values for uptrend, negative for downtrend)
        df['trend'] = df['close'].diff(5).rolling(window=5).mean()

        # Z-scores
        df['price_zscore'] = (df['close'] - df['close'].rolling(window=50).mean()) / df['close'].rolling(
            window=50).std()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=50).mean()) / df['volume'].rolling(
            window=50).std()

        # Detect anomalies based on z-scores
        df['is_anomaly'] = (abs(df['price_zscore']) > 3) | (abs(df['volume_zscore']) > 3)

        return df

    def make_stationary(self, data: pd.DataFrame, columns=None, method='diff',
                        order=1, seasonal_order=None) -> pd.DataFrame:
        """Перетворює часові ряди на стаціонарні для моделей ARIMA/SARIMA."""
        # (enhanced version of your method)
        if columns is None:
            columns = ['close']

        df = data.copy()
        original_df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            # Базове диференціювання
            if method == 'diff' or method == 'all':
                df[f'{col}_diff'] = df[col].diff(order)

                # Додаємо диференціювання 2-го порядку
                df[f'{col}_diff2'] = df[col].diff().diff()

                # Додаємо сезонне диференціювання якщо потрібно
                if seasonal_order:
                    df[f'{col}_seasonal_diff'] = df[col].diff(seasonal_order)
                    # Комбіноване сезонне + звичайне диференціювання
                    df[f'{col}_combo_diff'] = df[col].diff(seasonal_order).diff()

            # Логарифмічне перетворення
            if method == 'log' or method == 'all':
                # Перевірка на відсутність нульових чи від'ємних значень
                if (df[col] > 0).all():
                    df[f'{col}_log'] = np.log(df[col])
                    # Логарифм + диференціювання
                    df[f'{col}_log_diff'] = df[f'{col}_log'].diff(order)
                else:
                    self.logger.warning(
                        f"Колонка {col} містить нульові або від'ємні значення. Логарифмічне перетворення пропущено.")

            # Відсоткова зміна
            if method == 'pct_change' or method == 'all':
                df[f'{col}_pct'] = df[col].pct_change(order)

        # Видалення рядків з NaN після диференціювання
        df = df.dropna()

        # Зберігаємо оригінальні дані для можливості зворотного перетворення
        self.original_data = original_df

        return df

    def check_stationarity(self, data: pd.DataFrame, column='close') -> dict:
        """
        Перевіряє стаціонарність часового ряду за допомогою тестів.
        """
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

        results = {}

        # Тест Дікі-Фуллера (нульова гіпотеза: ряд нестаціонарний)
        adf_result = adfuller(data[column].dropna())
        results['adf_test'] = {
            'test_statistic': adf_result[0],
            'p-value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'critical_values': adf_result[4]
        }

        # KPSS тест (нульова гіпотеза: ряд стаціонарний)
        try:
            kpss_result = kpss(data[column].dropna())
            results['kpss_test'] = {
                'test_statistic': kpss_result[0],
                'p-value': kpss_result[1],
                'is_stationary': kpss_result[1] > 0.05,
                'critical_values': kpss_result[3]
            }
        except:
            self.logger.warning("Помилка при виконанні KPSS тесту")
            results['kpss_test'] = {
                'error': 'Failed to compute KPSS test'
            }

        # ACF and PACF для визначення параметрів моделі ARIMA
        try:
            acf_values = acf(data[column].dropna(), nlags=40)
            pacf_values = pacf(data[column].dropna(), nlags=40)

            # Знаходження значимих лагів (з 95% довірчим інтервалом)
            n = len(data[column].dropna())
            confidence_interval = 1.96 / np.sqrt(n)

            significant_acf_lags = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
            significant_pacf_lags = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]

            results['acf_pacf'] = {
                'significant_acf_lags': significant_acf_lags,
                'significant_pacf_lags': significant_pacf_lags,
                'suggested_p': min(significant_pacf_lags) if significant_pacf_lags else 0,
                'suggested_q': min(significant_acf_lags) if significant_acf_lags else 0
            }
        except:
            self.logger.warning("Помилка при розрахунку ACF/PACF")
            results['acf_pacf'] = {
                'error': 'Failed to compute ACF/PACF'
            }

        # Загальний висновок про стаціонарність
        results['is_stationary'] = results['adf_test']['is_stationary'] and \
                                   (results.get('kpss_test', {}).get('is_stationary', False) or
                                    'error' in results.get('kpss_test', {}))

        return results

    # New methods for database interactions

    def prepare_arima_data(self, data: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
        """Підготовка даних для ARIMA/SARIMA моделей і запис в базу."""
        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для підготовки ARIMA даних")
            return pd.DataFrame()

        self.logger.info(f"Підготовка ARIMA даних для {asset} на інтервалі {timeframe}")

        # 1. Перевірка стаціонарності вихідних даних
        stationarity_results = self.check_stationarity(data, 'close')

        # 2. Перетворення для стаціонарності
        stationary_data = self.make_stationary(data, columns=['close'], method='all')

        # 3. Перевірка стаціонарності після перетворень
        diff_stationarity = self.check_stationarity(stationary_data, 'close_diff')
        log_diff_stationarity = self.check_stationarity(stationary_data,
                                                        'close_log_diff') if 'close_log_diff' in stationary_data.columns else None

        # 4. Підготовка даних для запису в БД
        arima_data = pd.DataFrame()
        arima_data['open_time'] = data.index
        arima_data['original_close'] = data['close']

        # Додаємо трансформовані дані
        for col in ['close_diff', 'close_diff2', 'close_log', 'close_log_diff', 'close_pct',
                    'close_seasonal_diff', 'close_combo_diff']:
            if col in stationary_data.columns:
                arima_data[col] = stationary_data[col]

        # Додаємо результати тестів на стаціонарність
        arima_data['adf_pvalue'] = diff_stationarity['adf_test']['p-value']
        arima_data['kpss_pvalue'] = diff_stationarity.get('kpss_test', {}).get('p-value', None)
        arima_data['is_stationary'] = diff_stationarity['is_stationary']

        # Серіалізуємо значимі лаги для ACF/PACF
        if 'acf_pacf' in diff_stationarity and 'error' not in diff_stationarity['acf_pacf']:
            arima_data['significant_lags'] = arima_data.index.map(
                lambda _: json.dumps({
                    'acf': diff_stationarity['acf_pacf']['significant_acf_lags'],
                    'pacf': diff_stationarity['acf_pacf']['significant_pacf_lags']
                })
            )

        # Додаємо метадані
        arima_data['timeframe'] = timeframe

        # 5. Спроба підбору параметрів ARIMA для найкращого стаціонарного перетворення
        try:
            # Вибираємо найкраще перетворення на основі ADF p-value
            best_transform = 'close_diff'
            best_adf = diff_stationarity['adf_test']['p-value']

            if log_diff_stationarity and log_diff_stationarity['adf_test']['p-value'] < best_adf:
                best_transform = 'close_log_diff'
                best_adf = log_diff_stationarity['adf_test']['p-value']

            # Вибираємо параметри p, d, q на основі ACF/PACF
            p = diff_stationarity.get('acf_pacf', {}).get('suggested_p', 1)
            d = 1  # Ми вже диференціювали ряд
            q = diff_stationarity.get('acf_pacf', {}).get('suggested_q', 1)

            # Підганяємо ARIMA модель
            from statsmodels.tsa.arima.model import ARIMA

            # Обмежуємо значення параметрів для уникнення надто складних моделей
            p = min(p, 5)
            q = min(q, 5)

            model = ARIMA(stationary_data[best_transform].dropna(), order=(p, 0, q))
            model_fit = model.fit()

            # Додаємо метрики моделі
            arima_data['aic_score'] = model_fit.aic
            arima_data['bic_score'] = model_fit.bic
            arima_data['residual_variance'] = model_fit.resid.var()

            self.logger.info(f"ARIMA модель успішно створена для {asset} з параметрами ({p},{d},{q})")
        except Exception as e:
            self.logger.warning(f"Помилка при підборі ARIMA моделі: {str(e)}")

        return arima_data

    def prepare_lstm_data(self, data: pd.DataFrame, asset: str, timeframe: str,
                          sequence_length: int = 60, forecast_horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Підготовка даних для LSTM/GRU моделей і запис в базу.

        Параметри:
        - data: вихідні дані
        - asset: назва активу (btc, eth, sol)
        - timeframe: часовий інтервал
        - sequence_length: довжина послідовності для прогнозування
        - forecast_horizons: горизонти прогнозування (кількість кроків вперед)
        """
        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для підготовки LSTM даних")
            return pd.DataFrame()

        self.logger.info(f"Підготовка LSTM даних для {asset} на інтервалі {timeframe}")

        # 1. Додаємо технічні індикатори
        df = self.calculate_technical_indicators(data)

        # 2. Додаємо часові ознаки
        df = self.create_time_features(df)

        # 3. Вибір колонок для масштабування
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'ma_5', 'ma_20', 'rsi_14', 'macd',
            'bb_upper', 'bb_lower', 'volatility'
        ]

        # Додаємо циклічні часові ознаки
        time_features = [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos'
        ]

        # Перевіряємо наявність колонок
        available_features = [col for col in feature_cols if col in df.columns]
        available_time_features = [col for col in time_features if col in df.columns]

        # 4. Масштабування даних
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Масштабування ознак
        scaled_features = pd.DataFrame(
            scaler.fit_transform(df[available_features].fillna(method='ffill').fillna(0)),
            columns=[f"{col}_scaled" for col in available_features],
            index=df.index
        )

        # Зберігаємо скалер для відновлення даних
        scaler_key = f"{asset}_{timeframe}"
        self.scalers[scaler_key] = scaler

        # Зберігаємо параметри скалера
        scaling_metadata = {
            'feature_min_': dict(zip(available_features, scaler.min_.tolist())),
            'feature_scale_': dict(zip(available_features, scaler.scale_.tolist()))
        }

        # 5. Створення послідовностей для обробки
        sequences_df = pd.DataFrame()
        sequence_counter = 0

        # Для кожної можливої послідовності
        for i in range(len(df) - sequence_length - max(forecast_horizons) + 1):
            sequence_id = sequence_counter
            sequence_counter += 1

            # Ітеруємо по послідовності
            for pos in range(sequence_length):
                idx = i + pos
                row = {
                    'sequence_id': sequence_id,
                    'sequence_position': pos,
                    'open_time': df.index[idx],
                    'timeframe': timeframe,
                    'sequence_length': sequence_length
                }

                # Додаємо масштабовані ознаки
                for col in scaled_features.columns:
                    row[col] = scaled_features.iloc[idx][col]

                # Додаємо цільові значення для різних горизонтів
                for horizon in forecast_horizons:
                    if i + pos + horizon < len(df):
                        row[f'target_close_{horizon}'] = df['close'].iloc[i + pos + horizon]
                    else:
                        row[f'target_close_{horizon}'] = None

                # Додаємо часові ознаки без масштабування (вони вже нормалізовані)
                for feature in available_time_features:
                    row[feature] = df[feature].iloc[idx]

                # Додаємо метадані масштабування
                row['scaling_metadata'] = json.dumps(scaling_metadata)

                # Додаємо рядок до результуючого DataFrame
                sequences_df = pd.concat([sequences_df, pd.DataFrame([row])], ignore_index=True)

        self.logger.info(f"Створено {sequence_counter} послідовностей для {asset}")

        return sequences_df

    def save_arima_data_to_db(self, data: pd.DataFrame, asset: str):
        """Зберігає підготовлені дані для ARIMA/SARIMA моделей в базу даних."""
        if data.empty:
            self.logger.warning(f"Немає даних для збереження в {asset}_arima_data")
            return

        self.logger.info(f"Збереження {len(data)} рядків ARIMA даних для {asset}")

        table_name = f"{asset}_arima_data"

        try:
            # Перевіряємо чи існує таблиця
            table_exists = self.db_manager.execute_query(
                f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')",
                fetch_one=True
            )[0]

            if not table_exists:
                self.logger.info(f"Таблиця {table_name} не існує. Створення...")

                # Створюємо таблицю
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    timeframe TEXT NOT NULL,
                    open_time TIMESTAMP NOT NULL,

                    -- Original data (for reference)
                    original_close NUMERIC NOT NULL,

                    -- Stationary transformations
                    close_diff NUMERIC,
                    close_diff2 NUMERIC,
                    close_log NUMERIC,
                    close_log_diff NUMERIC,
                    close_pct_change NUMERIC,

                    -- Seasonal differencing
                    close_seasonal_diff NUMERIC,
                    close_combo_diff NUMERIC,

                    -- Stationarity test results
                    adf_pvalue NUMERIC,
                    kpss_pvalue NUMERIC,
                    is_stationary BOOLEAN,

                    -- ACF/PACF info
                    significant_lags TEXT,

                    -- Additional ARIMA-specific metrics
                    residual_variance NUMERIC,
                    aic_score NUMERIC,
                    bic_score NUMERIC,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (timeframe, open_time)
                );
                """
                self.db_manager.execute_query(create_table_query)
                self.logger.info(f"Таблиця {table_name} створена успішно")

            # Підготовка даних для запису
            columns = [col for col in data.columns if col not in ['id', 'created_at', 'updated_at']]
            placeholders = [f"%({col})s" for col in columns]

            insert_query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (timeframe, open_time) 
            DO UPDATE SET 
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['timeframe', 'open_time']])},
                updated_at = CURRENT_TIMESTAMP
            """

            # Перетворення DataFrame на список словників для запису
            records = data.to_dict('records')

            # Запис даних пакетами
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.db_manager.execute_batch(insert_query, batch)
                self.logger.info(f"Записано пакет {i // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

            self.logger.info(f"Дані для {asset} успішно збережені в таблицю {table_name}")

        except Exception as e:
            self.logger.error(f"Помилка при збереженні ARIMA даних для {asset}: {str(e)}")
            raise

    def save_lstm_data_to_db(self, data: pd.DataFrame, asset: str):
        """Зберігає підготовлені дані для LSTM/GRU моделей в базу даних."""
        if data.empty:
            self.logger.warning(f"Немає даних для збереження в {asset}_lstm_data")
            return

        self.logger.info(f"Збереження {len(data)} рядків LSTM даних для {asset}")

        table_name = f"{asset}_lstm_data"

        try:
            # Перевіряємо чи існує таблиця
            table_exists = self.db_manager.execute_query(
                f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')",
                fetch_one=True
            )[0]

            if not table_exists:
                self.logger.info(f"Таблиця {table_name} не існує. Створення...")

                # Створюємо таблицю з динамічними колонками на основі даних
                columns = []

                # Додаємо базові колонки
                columns.append("id SERIAL PRIMARY KEY")
                columns.append("timeframe TEXT NOT NULL")
                columns.append("sequence_id INTEGER NOT NULL")
                columns.append("sequence_position INTEGER NOT NULL")
                columns.append("open_time TIMESTAMP NOT NULL")
                columns.append("sequence_length INTEGER")

                # Додаємо колонки для ознак (виключаємо метадані і службові поля)
                exclude_cols = ['id', 'sequence_id', 'sequence_position', 'open_time', 'timeframe',
                                'sequence_length', 'scaling_metadata', 'created_at', 'updated_at']

                for col in data.columns:
                    if col not in exclude_cols:
                        if 'target' in col:
                            columns.append(f"{col} NUMERIC")
                        elif col == 'scaling_metadata':
                            columns.append(f"{col} TEXT")
                        else:
                            columns.append(f"{col} NUMERIC NOT NULL")

                # Додаємо метадані і унікальні обмеження
                columns.append("scaling_metadata TEXT")
                columns.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                columns.append("updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                columns.append("UNIQUE (timeframe, sequence_id, sequence_position)")

                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns)}
                );
                """
                self.db_manager.execute_query(create_table_query)
                self.logger.info(f"Таблиця {table_name} створена успішно")

            # Підготовка даних для запису
            columns = [col for col in data.columns if col not in ['id', 'created_at', 'updated_at']]
            placeholders = [f"%({col})s" for col in columns]

            insert_query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (timeframe, sequence_id, sequence_position) 
            DO UPDATE SET 
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['timeframe', 'sequence_id', 'sequence_position']])},
                updated_at = CURRENT_TIMESTAMP
            """

            # Перетворення DataFrame на список словників для запису
            records = data.to_dict('records')

            # Запис даних пакетами
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.db_manager.execute_batch(insert_query, batch)
                self.logger.info(f"Записано пакет {i // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

            self.logger.info(f"Дані для {asset} успішно збережені в таблицю {table_name}")

        except Exception as e:
            self.logger.error(f"Помилка при збереженні LSTM даних для {asset}: {str(e)}")
            raise

    def process_data_for_all_models(self, data: pd.DataFrame, asset: str, timeframe: str):
        """Обробка даних для всіх типів моделей і збереження в базу."""
        self.logger.info(f"Початок обробки даних для всіх моделей: {asset}, {timeframe}")

        try:
            # 1. Ресемплінг даних, якщо потрібно
            # (Припускаємо, що дані вже ресемпльовані до потрібного часового інтервалу)

            # 2. Підготовка даних для ARIMA/SARIMA
            arima_data = self.prepare_arima_data(data, asset, timeframe)
            self.save_arima_data_to_db(arima_data, asset)

            # 3. Підготовка даних для LSTM/GRU
            lstm_data = self.prepare_lstm_data(data, asset, timeframe)
            self.save_lstm_data_to_db(lstm_data, asset)

            self.logger.info(f"Обробка даних для всіх моделей завершена: {asset}, {timeframe}")

            return {
                'arima_rows': len(arima_data),
                'lstm_rows': len(lstm_data)
            }

        except Exception as e:
            self.logger.error(f"Помилка при обробці даних для моделей: {str(e)}")
            raise