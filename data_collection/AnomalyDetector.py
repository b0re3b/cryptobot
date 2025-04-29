from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


class AnomalyDetector:
    def __init__(self, logger):
        self.logger = logger

    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                        threshold: float = 3) -> Tuple[pd.DataFrame, List]:

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для виявлення аномалій")
            return pd.DataFrame(), []

        self.logger.info(f"Початок виявлення аномалій методом {method} з порогом {threshold}")

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("У DataFrame відсутні числові колонки для аналізу аномалій")
            return pd.DataFrame(), []

        outliers_df = pd.DataFrame(index=data.index)
        all_outlier_indices = set()

        if method == 'zscore':
            for col in numeric_cols:
                # Перевірка на стандартне відхилення == 0
                std = data[col].std()
                if std == 0 or pd.isna(std):
                    self.logger.warning(f"Колонка {col} має нульове стандартне відхилення або NaN")
                    continue

                z_scores = np.abs((data[col] - data[col].mean()) / std)
                outliers = z_scores > threshold
                outliers_df[f'{col}_outlier'] = outliers

                if outliers.any():
                    self.logger.info(f"Знайдено {outliers.sum()} аномалій у колонці {col} (zscore)")
                    all_outlier_indices.update(data.index[outliers])

        elif method == 'iqr':
            for col in numeric_cols:
                # Перевірка на достатню кількість даних для обчислення квартилів
                if len(data[col].dropna()) < 4:
                    self.logger.warning(f"Недостатньо даних у колонці {col} для IQR методу")
                    continue

                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0 or pd.isna(IQR):
                    self.logger.warning(f"Колонка {col} має нульовий IQR або NaN")
                    continue

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers_df[f'{col}_outlier'] = outliers

                if outliers.any():
                    self.logger.info(f"Знайдено {outliers.sum()} аномалій у колонці {col} (IQR)")
                    all_outlier_indices.update(data.index[outliers])

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest

                # Перевірка наявності достатньої кількості даних
                if len(data) < 10:
                    self.logger.warning("Недостатньо даних для Isolation Forest")
                    return pd.DataFrame(), []

                X = data[numeric_cols].fillna(data[numeric_cols].mean())

                # Перевірка на наявність NaN після заповнення
                if X.isna().any().any():
                    self.logger.warning("Залишились NaN після заповнення. Вони будуть замінені на 0")
                    X = X.fillna(0)

                model = IsolationForest(contamination=min(0.1, 1 / threshold), random_state=42)
                predictions = model.fit_predict(X)

                outliers = predictions == -1

                outliers_df['isolation_forest_outlier'] = outliers

                if outliers.any():
                    self.logger.info(f"Знайдено {outliers.sum()} аномалій методом Isolation Forest")
                    all_outlier_indices.update(data.index[outliers])

            except ImportError:
                self.logger.error("Для використання методу 'isolation_forest' необхідно встановити scikit-learn")
                return pd.DataFrame(), []

        else:
            self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
            return pd.DataFrame(), []

        if not outliers_df.empty:
            outliers_df['is_outlier'] = outliers_df.any(axis=1)

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return outliers_df, outlier_indices

    def validate_data_integrity(self, data: pd.DataFrame, price_jump_threshold: float = 0.2,
                                volume_anomaly_threshold: float = 5) -> Dict[str, List]:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для перевірки цілісності")
            return {"empty_data": []}

        issues = {}

        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            issues["missing_columns"] = missing_cols
            self.logger.warning(f"Відсутні колонки: {missing_cols}")

        if not isinstance(data.index, pd.DatetimeIndex):
            issues["not_datetime_index"] = True
            self.logger.warning("Індекс не є DatetimeIndex")
        else:
            # Перевірка часових проміжків
            if len(data.index) > 1:
                time_diff = data.index.to_series().diff().dropna()
                if not time_diff.empty:
                    median_diff = time_diff.median()

                    # Перевірка на великі проміжки
                    if median_diff.total_seconds() > 0:  # Уникаємо ділення на нуль
                        large_gaps = time_diff[time_diff > 2 * median_diff]
                        if not large_gaps.empty:
                            gap_locations = large_gaps.index.tolist()
                            issues["time_gaps"] = gap_locations
                            self.logger.warning(f"Знайдено {len(gap_locations)} аномальних проміжків у часових мітках")

                    # Перевірка на дублікати часових міток
                    duplicates = data.index.duplicated()
                    if duplicates.any():
                        dup_indices = data.index[duplicates].tolist()
                        issues["duplicate_timestamps"] = dup_indices
                        self.logger.warning(f"Знайдено {len(dup_indices)} дублікатів часових міток")

        # Перевірка цінових аномалій
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        if len(price_cols) == 4:
            # Перевірка high < low
            invalid_hl = data['high'] < data['low']
            if invalid_hl.any():
                invalid_hl_indices = data.index[invalid_hl].tolist()
                issues["high_lower_than_low"] = invalid_hl_indices
                self.logger.warning(f"Знайдено {len(invalid_hl_indices)} записів де high < low")

            # Перевірка від'ємних цін
            for col in price_cols:
                negative_prices = data[col] < 0
                if negative_prices.any():
                    neg_price_indices = data.index[negative_prices].tolist()
                    issues[f"negative_{col}"] = neg_price_indices
                    self.logger.warning(f"Знайдено {len(neg_price_indices)} записів з від'ємними значеннями у {col}")

            # Перевірка різких стрибків цін
            for col in price_cols:
                pct_change = data[col].pct_change().abs()
                price_jumps = pct_change > price_jump_threshold
                if price_jumps.any():
                    jump_indices = data.index[price_jumps].tolist()
                    issues[f"price_jumps_{col}"] = jump_indices
                    self.logger.warning(f"Знайдено {len(jump_indices)} різких змін у колонці {col}")

        # Перевірка об'єму
        if 'volume' in data.columns:
            # Перевірка від'ємного об'єму
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                neg_vol_indices = data.index[negative_volume].tolist()
                issues["negative_volume"] = neg_vol_indices
                self.logger.warning(f"Знайдено {len(neg_vol_indices)} записів з від'ємним об'ємом")

            # Перевірка аномального об'єму
            try:
                volume_std = data['volume'].std()
                if volume_std > 0:  # Уникаємо ділення на нуль
                    volume_zscore = np.abs((data['volume'] - data['volume'].mean()) / volume_std)
                    volume_anomalies = volume_zscore > volume_anomaly_threshold
                    if volume_anomalies.any():
                        vol_anomaly_indices = data.index[volume_anomalies].tolist()
                        issues["volume_anomalies"] = vol_anomaly_indices
                        self.logger.warning(f"Знайдено {len(vol_anomaly_indices)} записів з аномальним об'ємом")
            except Exception as e:
                self.logger.error(f"Помилка при аналізі аномалій об'єму: {str(e)}")

        # Перевірка на NaN значення
        na_counts = data.isna().sum()
        cols_with_na = na_counts[na_counts > 0].index.tolist()
        if cols_with_na:
            issues["columns_with_na"] = {col: data.index[data[col].isna()].tolist() for col in cols_with_na}
            self.logger.warning(f"Знайдено відсутні значення у колонках: {cols_with_na}")

        # Перевірка на нескінченні значення
        try:
            inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
            cols_with_inf = inf_counts[inf_counts > 0].index.tolist()
            if cols_with_inf:
                issues["columns_with_inf"] = {col: data.index[np.isinf(data[col])].tolist() for col in cols_with_inf}
                self.logger.warning(f"Знайдено нескінченні значення у колонках: {cols_with_inf}")
        except Exception as e:
            self.logger.error(f"Помилка при перевірці нескінченних значень: {str(e)}")

        return issues