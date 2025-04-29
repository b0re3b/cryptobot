from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd


class AnomalyDetector:

    def __init__(self, logger: Any):

        self.logger = logger

    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                        threshold: float = 3) -> Tuple[pd.DataFrame, List]:
        """
        Виявляє аномалії у даних за допомогою вказаного методу.

        Args:
            data: DataFrame з даними для аналізу
            method: Метод виявлення аномалій ('zscore', 'iqr', 'isolation_forest')
            threshold: Поріг для визначення аномалій

        Returns:
            Tuple з DataFrame, що містить інформацію про аномалії, та списком індексів аномальних записів
        """
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для виявлення аномалій")
            return pd.DataFrame(), []

        if threshold <= 0:
            self.logger.warning(f"Отримано неприпустимий поріг: {threshold}. Встановлено значення за замовчуванням 3")
            threshold = 3

        self.logger.info(f"Початок виявлення аномалій методом {method} з порогом {threshold}")

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("У DataFrame відсутні числові колонки для аналізу аномалій")
            return pd.DataFrame(), []

        outliers_df = pd.DataFrame(index=data.index)
        all_outlier_indices = set()

        try:
            if method == 'zscore':
                self._detect_zscore_outliers(data, numeric_cols, threshold, outliers_df, all_outlier_indices)
            elif method == 'iqr':
                self._detect_iqr_outliers(data, numeric_cols, threshold, outliers_df, all_outlier_indices)
            elif method == 'isolation_forest':
                self._detect_isolation_forest_outliers(data, numeric_cols, threshold, outliers_df, all_outlier_indices)
            else:
                self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
                return pd.DataFrame(), []

        except Exception as e:
            self.logger.error(f"Помилка при виявленні аномалій методом {method}: {str(e)}")
            return pd.DataFrame(), []

        if not outliers_df.empty:
            outliers_df['is_outlier'] = outliers_df.any(axis=1)

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return outliers_df, outlier_indices

    def _detect_zscore_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                                threshold: float, outliers_df: pd.DataFrame,
                                all_outlier_indices: set) -> None:
        """
        Допоміжний метод для виявлення аномалій за допомогою z-score.
        """
        for col in numeric_cols:
            # Пропускаємо колонки з NaN значеннями
            if data[col].isna().all():
                self.logger.warning(f"Колонка {col} містить лише NaN значення, пропускаємо")
                continue

            # Працюємо з не-NaN значеннями
            valid_data = data[col].dropna()
            if valid_data.empty:
                continue

            # Перевірка на стандартне відхилення == 0
            std = valid_data.std()
            if std == 0 or pd.isna(std):
                self.logger.warning(f"Колонка {col} має нульове стандартне відхилення або NaN")
                continue

            # Розраховуємо z-score тільки для дійсних значень
            mean = valid_data.mean()
            z_scores = pd.Series(index=data.index, dtype=float)
            z_scores[valid_data.index] = np.abs((valid_data - mean) / std)

            # Позначаємо аномалії
            outliers = z_scores > threshold
            outliers_df[f'{col}_outlier'] = outliers

            if outliers.any():
                outlier_count = outliers.sum()
                self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (zscore)")
                all_outlier_indices.update(data.index[outliers])

    def _detect_iqr_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                             threshold: float, outliers_df: pd.DataFrame,
                             all_outlier_indices: set) -> None:
        """
        Допоміжний метод для виявлення аномалій за допомогою IQR.
        """
        for col in numeric_cols:
            # Пропускаємо колонки з NaN значеннями
            valid_data = data[col].dropna()
            if len(valid_data) < 4:
                self.logger.warning(f"Недостатньо даних у колонці {col} для IQR методу")
                continue

            try:
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1

                if IQR <= 0 or pd.isna(IQR):
                    self.logger.warning(f"Колонка {col} має нульовий або від'ємний IQR або NaN")
                    continue

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Створюємо серію з індексами original data
                outliers = pd.Series(False, index=data.index)
                outliers[valid_data.index] = (valid_data < lower_bound) | (valid_data > upper_bound)
                outliers_df[f'{col}_outlier'] = outliers

                if outliers.any():
                    outlier_count = outliers.sum()
                    self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (IQR)")
                    all_outlier_indices.update(data.index[outliers])

            except Exception as e:
                self.logger.error(f"Помилка при застосуванні IQR методу до колонки {col}: {str(e)}")

    def _detect_isolation_forest_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                                          threshold: float, outliers_df: pd.DataFrame,
                                          all_outlier_indices: set) -> None:
        """
        Допоміжний метод для виявлення аномалій за допомогою Isolation Forest.
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            self.logger.error("Для використання методу 'isolation_forest' необхідно встановити scikit-learn")
            return

        # Перевірка наявності достатньої кількості даних
        if len(data) < 10:
            self.logger.warning("Недостатньо даних для Isolation Forest (потрібно мінімум 10 записів)")
            return

        # Підготовка даних - заповнення NaN середніми значеннями
        X = data[numeric_cols].copy()
        for col in numeric_cols:
            if X[col].isna().any():
                col_mean = X[col].mean()
                if pd.isna(col_mean):  # Якщо середнє також NaN
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(col_mean)

        # Перевірка на наявність NaN після заповнення
        if X.isna().any().any():
            self.logger.warning("Залишились NaN після заповнення. Вони будуть замінені на 0")
            X = X.fillna(0)

        # Безпечне встановлення параметра contamination
        contamination = 0.1  # За замовчуванням
        if threshold > 0:
            contamination = min(0.1, 1 / threshold)
        else:
            self.logger.warning("Threshold має бути більше 0 для методу isolation_forest. "
                                "Використовуємо значення за замовчуванням 0.1")

        try:
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(X)

            outliers = predictions == -1
            outliers_df['isolation_forest_outlier'] = outliers

            if outliers.any():
                outlier_count = outliers.sum()
                self.logger.info(f"Знайдено {outlier_count} аномалій методом Isolation Forest")
                all_outlier_indices.update(data.index[outliers])

        except Exception as e:
            self.logger.error(f"Помилка при використанні Isolation Forest: {str(e)}")

    def validate_data_integrity(self, data: pd.DataFrame, price_jump_threshold: float = 0.2,
                                volume_anomaly_threshold: float = 5) -> Dict[str, Any]:
        """
        Перевіряє цілісність і коректність даних.

        Args:
            data: DataFrame з даними для перевірки
            price_jump_threshold: Поріг для виявлення різких змін цін
            volume_anomaly_threshold: Поріг для виявлення аномалій об'єму

        Returns:
            Словник з інформацією про знайдені проблеми
        """
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для перевірки цілісності")
            return {"empty_data": True}

        issues = {}

        # Перевірка параметрів
        if price_jump_threshold <= 0:
            self.logger.warning(f"Неприпустимий поріг для стрибків цін: {price_jump_threshold}. "
                                "Встановлено значення 0.2")
            price_jump_threshold = 0.2

        if volume_anomaly_threshold <= 0:
            self.logger.warning(f"Неприпустимий поріг для аномалій об'єму: {volume_anomaly_threshold}. "
                                "Встановлено значення 5")
            volume_anomaly_threshold = 5

        # Перевірка наявності очікуваних колонок
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            issues["missing_columns"] = missing_cols
            self.logger.warning(f"Відсутні колонки: {missing_cols}")

        # Перевірка індексу DataFrame
        try:
            self._validate_datetime_index(data, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці часового індексу: {str(e)}")
            issues["datetime_index_error"] = str(e)

        # Перевірка цінових даних
        try:
            self._validate_price_data(data, price_jump_threshold, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці цінових даних: {str(e)}")
            issues["price_validation_error"] = str(e)

        # Перевірка даних об'єму
        try:
            self._validate_volume_data(data, volume_anomaly_threshold, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці даних об'єму: {str(e)}")
            issues["volume_validation_error"] = str(e)

        # Перевірка на NaN і infinite значення
        try:
            self._validate_data_values(data, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці значень даних: {str(e)}")
            issues["data_values_error"] = str(e)

        return issues

    def _validate_datetime_index(self, data: pd.DataFrame, issues: Dict[str, Any]) -> None:
        """
        Допоміжний метод для перевірки часового індексу.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            issues["not_datetime_index"] = True
            self.logger.warning("Індекс не є DatetimeIndex")
            return

        # Перевірка часових проміжків
        if len(data.index) > 1:
            time_diff = data.index.to_series().diff().dropna()
            if not time_diff.empty:
                try:
                    median_diff = time_diff.median()

                    # Перевірка на можливість виклику total_seconds()
                    if hasattr(median_diff, 'total_seconds') and callable(getattr(median_diff, 'total_seconds')):
                        seconds = median_diff.total_seconds()
                        if seconds > 0:  # Уникаємо ділення на нуль
                            large_gaps = time_diff[time_diff > 2 * median_diff]
                            if not large_gaps.empty:
                                gap_locations = large_gaps.index.tolist()
                                issues["time_gaps"] = gap_locations
                                self.logger.warning(
                                    f"Знайдено {len(gap_locations)} аномальних проміжків у часових мітках")
                    else:
                        self.logger.warning("Неможливо обчислити часові проміжки між записами")
                except Exception as e:
                    self.logger.error(f"Помилка при аналізі часових проміжків: {str(e)}")

            # Перевірка на дублікати часових міток
            duplicates = data.index.duplicated()
            if duplicates.any():
                dup_indices = data.index[duplicates].tolist()
                issues["duplicate_timestamps"] = dup_indices
                self.logger.warning(f"Знайдено {len(dup_indices)} дублікатів часових міток")

    def _validate_price_data(self, data: pd.DataFrame, price_jump_threshold: float,
                             issues: Dict[str, Any]) -> None:
        """
        Допоміжний метод для перевірки цінових даних.
        """
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
                try:
                    # Безпечне обчислення відсоткової зміни з обробкою NaN
                    valid_data = data[col].dropna()
                    if len(valid_data) > 1:
                        pct_change = valid_data.pct_change().abs()
                        price_jumps = pct_change > price_jump_threshold
                        if price_jumps.any():
                            jump_indices = pct_change.index[price_jumps].tolist()
                            issues[f"price_jumps_{col}"] = jump_indices
                            self.logger.warning(f"Знайдено {len(jump_indices)} різких змін у колонці {col}")
                except Exception as e:
                    self.logger.error(f"Помилка при аналізі стрибків цін у колонці {col}: {str(e)}")

    def _validate_volume_data(self, data: pd.DataFrame, volume_anomaly_threshold: float,
                              issues: Dict[str, Any]) -> None:
        """
        Допоміжний метод для перевірки даних об'єму.
        """
        if 'volume' in data.columns:
            # Перевірка від'ємного об'єму
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                neg_vol_indices = data.index[negative_volume].tolist()
                issues["negative_volume"] = neg_vol_indices
                self.logger.warning(f"Знайдено {len(neg_vol_indices)} записів з від'ємним об'ємом")

            # Перевірка аномального об'єму
            try:
                valid_volume = data['volume'].dropna()
                if not valid_volume.empty:
                    volume_std = valid_volume.std()
                    volume_mean = valid_volume.mean()

                    if volume_std > 0:  # Уникаємо ділення на нуль
                        volume_zscore = np.abs((valid_volume - volume_mean) / volume_std)
                        volume_anomalies = volume_zscore > volume_anomaly_threshold

                        if volume_anomalies.any():
                            vol_anomaly_indices = valid_volume.index[volume_anomalies].tolist()
                            issues["volume_anomalies"] = vol_anomaly_indices
                            self.logger.warning(f"Знайдено {len(vol_anomaly_indices)} записів з аномальним об'ємом")
            except Exception as e:
                self.logger.error(f"Помилка при аналізі аномалій об'єму: {str(e)}")

    def _validate_data_values(self, data: pd.DataFrame, issues: Dict[str, Any]) -> None:
        """
        Допоміжний метод для перевірки значень даних (NaN, Inf).
        """
        # Перевірка на NaN значення
        na_counts = data.isna().sum()
        cols_with_na = na_counts[na_counts > 0].index.tolist()
        if cols_with_na:
            issues["columns_with_na"] = {col: data.index[data[col].isna()].tolist() for col in cols_with_na}
            self.logger.warning(f"Знайдено відсутні значення у колонках: {cols_with_na}")

        # Перевірка на нескінченні значення
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                inf_data = pd.DataFrame(index=data.index)
                for col in numeric_cols:
                    inf_data[col] = np.isinf(data[col])

                inf_counts = inf_data.sum()
                cols_with_inf = inf_counts[inf_counts > 0].index.tolist()

                if cols_with_inf:
                    issues["columns_with_inf"] = {col: data.index[np.isinf(data[col])].tolist() for col in
                                                  cols_with_inf}
                    self.logger.warning(f"Знайдено нескінченні значення у колонках: {cols_with_inf}")
        except Exception as e:
            self.logger.error(f"Помилка при перевірці нескінченних значень: {str(e)}")