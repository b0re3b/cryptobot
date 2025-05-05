from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import decimal

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnomalyDetector:

    def __init__(self, logger: Any):
        self.logger = logger

    def _ensure_float(self, df: pd.DataFrame) -> pd.DataFrame:

        result = df.copy()
        for col in result.columns:
            if result[col].dtype == object:
                has_decimal = any(isinstance(x, decimal.Decimal) for x in result[col].dropna())
                if has_decimal:
                    self.logger.info(f"Converting decimal.Decimal values to float in column {col}")
                    result[col] = result[col].apply(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)
        return result

    def _preprocess_data(self, data: pd.DataFrame, numeric_cols: List[str], fill_method: str = 'mean') -> pd.DataFrame:
        if data.empty or not numeric_cols:
            self.logger.warning("Порожні вхідні дані або відсутні числові колонки")
            return data

        processed_data = self._ensure_float(data)

        # 1. Заповнення відсутніх значень
        for col in numeric_cols:
            if col not in processed_data.columns:
                continue

            if processed_data[col].isna().any():
                if fill_method == 'mean':
                    fill_value = processed_data[col].mean()
                elif fill_method == 'median':
                    fill_value = processed_data[col].median()
                elif fill_method == 'ffill':
                    processed_data[col] = processed_data[col].ffill()
                    continue
                elif fill_method == 'bfill':
                    processed_data[col] = processed_data[col].bfill()
                    continue
                else:
                    fill_value = 0  # Запасний варіант

                processed_data[col] = processed_data[col].fillna(fill_value)

        # 2. Логарифмічне перетворення для сильно скошених даних (за потребою)
        for col in numeric_cols:
            if (processed_data[col] > 0).all():  # Логарифм визначений лише для додатних значень
                skewness = processed_data[col].skew()
                if abs(skewness) > 1.0:  # Сильне скошення
                    processed_data[f'{col}_log'] = np.log1p(processed_data[col])
                    self.logger.info(f"Застосовано логарифмічне перетворення для {col} (скошеність={skewness:.2f})")

        # 3. Видалення дублікатів індексу (якщо DataFrame має DatetimeIndex)
        if isinstance(processed_data.index, pd.DatetimeIndex):
            duplicates = processed_data.index.duplicated()
            if duplicates.any():
                processed_data = processed_data[~duplicates]
                self.logger.info(f"Видалено {duplicates.sum()} дублікатів індексу")

        # 4. Сортування за індексом (якщо це часовий ряд)
        if isinstance(processed_data.index, pd.DatetimeIndex):
            processed_data = processed_data.sort_index()

        return processed_data

    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                        threshold: float = 3.0, preprocess: bool = True,
                        fill_method: str = 'mean', contamination: float = 0.1) -> Tuple[pd.DataFrame, List]:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для виявлення аномалій")
            return pd.DataFrame(), []

        # Валідація параметрів
        if method not in ['zscore', 'iqr', 'isolation_forest']:
            self.logger.warning(f"Непідтримуваний метод: {method}. Використовуємо 'zscore'")
            method = 'zscore'

        if threshold <= 0:
            self.logger.warning(
                f"Отримано недопустиме порогове значення: {threshold}. Встановлено значення за замовчуванням 3")
            threshold = 3

        if contamination <= 0 or contamination > 0.5:
            self.logger.warning(f"Неправильне значення contamination: {contamination}. Використовуємо 0.1")
            contamination = 0.1

        self.logger.info(f"Початок виявлення аномалій методом {method} з порогом {threshold}")

        data = self._ensure_float(data)

        # Вибір числових колонок
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("У DataFrame немає числових колонок для аналізу аномалій")
            return pd.DataFrame(), []

        # Попередня обробка даних
        processed_data = data
        if preprocess:
            try:
                processed_data = self._preprocess_data(data, numeric_cols, fill_method)
                self.logger.info(f"Дані передоброблені з методом заповнення '{fill_method}'")
            except Exception as e:
                self.logger.error(f"Помилка під час передобробки даних: {str(e)}")
                processed_data = data

        outliers_df = pd.DataFrame(index=data.index)
        all_outlier_indices = set()

        try:
            if method == 'zscore':
                self._detect_zscore_outliers(processed_data, numeric_cols, threshold, outliers_df, all_outlier_indices)
            elif method == 'iqr':
                self._detect_iqr_outliers(processed_data, numeric_cols, threshold, outliers_df, all_outlier_indices)
            elif method == 'isolation_forest':
                # Для isolation_forest використовуємо contamination замість threshold
                self._detect_isolation_forest_outliers(processed_data, numeric_cols, contamination, outliers_df,
                                                       all_outlier_indices)
            else:
                self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
                return pd.DataFrame(), []

        except Exception as e:
            self.logger.error(f"Помилка під час виявлення аномалій методом {method}: {str(e)}")
            return pd.DataFrame(), []

        if not outliers_df.empty:
            outliers_df['is_outlier'] = outliers_df.any(axis=1)

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return outliers_df, outlier_indices

    def _detect_zscore_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                                threshold: float, outliers_df: pd.DataFrame,
                                all_outlier_indices: set) -> None:

        # Вектеризована обробка для швидкодії
        valid_cols = []

        # Відфільтруємо колонки з проблемами
        for col in numeric_cols:
            if data[col].isna().all():
                self.logger.warning(f"Колонка {col} містить лише NaN значення, пропускаємо")
                continue

            valid_data = data[col].dropna()
            if valid_data.empty:
                continue

            valid_data = pd.to_numeric(valid_data, errors='coerce')

            std = valid_data.std()
            if std == 0 or pd.isna(std):
                self.logger.warning(f"Колонка {col} має нульове стандартне відхилення або NaN")
                continue

            valid_cols.append(col)

        if not valid_cols:
            self.logger.warning("Немає валідних колонок для Z-Score аналізу")
            return

        # Обчислимо Z-Score для всіх валідних колонок одночасно
        for col in valid_cols:
            valid_data = data[col].dropna()
            valid_data = pd.to_numeric(valid_data, errors='coerce')

            mean = valid_data.mean()
            std = valid_data.std()

            # Ініціалізуємо серію для Z-Score з NaN
            z_scores = pd.Series(np.nan, index=data.index)

            # Заповнюємо тільки валідні індекси
            z_scores[valid_data.index] = np.abs((valid_data - mean) / std)

            # Визначаємо аномалії
            outliers = z_scores > threshold
            outliers_df[f'{col}_outlier'] = outliers

            if outliers.any():
                outlier_count = outliers.sum()
                self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (zscore)")
                all_outlier_indices.update(data.index[outliers])

    def _detect_iqr_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                             threshold: float, outliers_df: pd.DataFrame,
                             all_outlier_indices: set) -> None:
        for col in numeric_cols:
            valid_data = data[col].dropna()
            if len(valid_data) < 4:
                self.logger.warning(f"Недостатньо даних у колонці {col} для IQR методу")
                continue

            valid_data = pd.to_numeric(valid_data, errors='coerce')

            try:
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1

                # Покращено перевірку крайнього випадку
                if Q1 == Q3:
                    self.logger.warning(f"Колонка {col} має однакові значення Q1 та Q3 (всі значення однакові)")
                    continue

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
                                          contamination: float, outliers_df: pd.DataFrame,
                                          all_outlier_indices: set) -> None:

        if not SKLEARN_AVAILABLE:
            self.logger.error("Для використання методу 'isolation_forest' необхідно встановити scikit-learn")
            return

        # Перевірка наявності достатньої кількості даних
        if len(data) < 10:
            self.logger.warning("Недостатньо даних для Isolation Forest (потрібно мінімум 10 записів)")
            return

        # Підготовка даних - виділення тільки числових колонок без NaN
        X = data[numeric_cols].copy()

        # Перевірка на однорідність даних
        if all(X[col].nunique() <= 1 for col in X.columns):
            self.logger.warning("Дані однорідні для всіх колонок, Isolation Forest буде неефективним")
            return

        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Заповнення пропущених значень
        for col in numeric_cols:
            if col in X.columns and X[col].isna().any():
                col_mean = X[col].mean()
                if pd.isna(col_mean):  # Якщо середнє також NaN
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(col_mean)

        # Перевірка на наявність NaN після заповнення
        if X.isna().any().any():
            self.logger.warning("Залишились NaN після заповнення. Вони будуть замінені на 0")
            X = X.fillna(0)

        # Валідація параметра contamination
        if contamination <= 0 or contamination > 0.5:
            self.logger.warning(f"Неправильне значення contamination: {contamination}. Використовуємо 0.1")
            contamination = 0.1

        try:
            model = IsolationForest(contamination=contamination,
                                    random_state=42,
                                    n_estimators=100,
                                    max_samples='auto')
            predictions = model.fit_predict(X)

            # -1 для викидів, 1 для нормальних значень
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

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для перевірки цілісності")
            return {"empty_data": True}

        data = self._ensure_float(data)

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

        if not isinstance(data.index, pd.DatetimeIndex):
            issues["not_datetime_index"] = True
            self.logger.warning("Індекс не є DatetimeIndex")
            return

        # Перевірка на впорядкованість дат
        if not data.index.is_monotonic_increasing:
            issues["unordered_timestamps"] = True
            self.logger.warning("Часові мітки не впорядковані за зростанням")

            # Визначаємо місця, де порушується порядок
            if len(data.index) > 1:
                unordered_locations = []
                for i in range(1, len(data.index)):
                    if data.index[i] < data.index[i - 1]:
                        unordered_locations.append((i - 1, i, data.index[i - 1], data.index[i]))

                if unordered_locations:
                    issues["unordered_locations"] = unordered_locations[:10]  # Обмежуємо кількість записів

        # Перевірка часових проміжків
        if len(data.index) > 1:
            time_diff = data.index.to_series().diff().dropna()
            if not time_diff.empty:
                try:
                    # Розрахунок статистики часових проміжків
                    median_diff = time_diff.median()
                    mean_diff = time_diff.mean()
                    min_diff = time_diff.min()
                    max_diff = time_diff.max()

                    # Записуємо статистику
                    issues["time_diff_stats"] = {
                        "median_seconds": median_diff.total_seconds() if hasattr(median_diff,
                                                                                 'total_seconds') else None,
                        "mean_seconds": mean_diff.total_seconds() if hasattr(mean_diff, 'total_seconds') else None,
                        "min_seconds": min_diff.total_seconds() if hasattr(min_diff, 'total_seconds') else None,
                        "max_seconds": max_diff.total_seconds() if hasattr(max_diff, 'total_seconds') else None
                    }

                    # Перевірка на аномальні проміжки
                    if hasattr(median_diff, 'total_seconds') and callable(getattr(median_diff, 'total_seconds')):
                        seconds = median_diff.total_seconds()
                        if seconds > 0:  # Уникаємо ділення на нуль
                            # Виявляємо великі проміжки (в 2+ рази більші за медіану)
                            large_gaps = time_diff[time_diff > 2 * median_diff]
                            if not large_gaps.empty:
                                gap_locations = []
                                for idx in large_gaps.index:
                                    gap_locations.append({
                                        "timestamp": str(idx),
                                        "previous_timestamp": str(idx - time_diff[idx]),
                                        "gap_seconds": time_diff[idx].total_seconds()
                                    })

                                issues["time_gaps"] = gap_locations
                                self.logger.warning(
                                    f"Знайдено {len(gap_locations)} аномальних проміжків у часових мітках")

                            # Виявляємо аномально малі проміжки
                            small_gaps = time_diff[time_diff < median_diff * 0.1]
                            if not small_gaps.empty and not np.isclose(median_diff.total_seconds(), 0):
                                small_gap_locations = []
                                for idx in small_gaps.index:
                                    small_gap_locations.append({
                                        "timestamp": str(idx),
                                        "previous_timestamp": str(idx - time_diff[idx]),
                                        "gap_seconds": time_diff[idx].total_seconds()
                                    })

                                issues["small_time_gaps"] = small_gap_locations
                                self.logger.warning(
                                    f"Знайдено {len(small_gap_locations)} аномально малих проміжків у часових мітках")
                    else:
                        self.logger.warning("Неможливо обчислити часові проміжки між записами")
                except Exception as e:
                    self.logger.error(f"Помилка при аналізі часових проміжків: {str(e)}")

            # Перевірка на дублікати часових міток
            duplicates = data.index.duplicated()
            if duplicates.any():
                dup_indices = data.index[duplicates].tolist()
                issues["duplicate_timestamps"] = [str(ts) for ts in dup_indices[:20]]  # Обмежуємо до 20 записів
                self.logger.warning(f"Знайдено {duplicates.sum()} дублікатів часових міток")

            # Перевірка на часову частоту
            try:
                inferred_freq = pd.infer_freq(data.index)
                issues["inferred_frequency"] = inferred_freq
                if inferred_freq is None:
                    self.logger.info("Неможливо визначити регулярну частоту часового ряду")
                else:
                    self.logger.info(f"Визначена частота часового ряду: {inferred_freq}")
            except Exception as e:
                self.logger.error(f"Помилка при визначенні частоти часового ряду: {str(e)}")

    def _validate_price_data(self, data: pd.DataFrame, price_jump_threshold: float,
                             issues: Dict[str, Any]) -> None:

        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]

        for col in price_cols:
            if data[col].dtype == object:
                data[col] = pd.to_numeric(data[col], errors='coerce')

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
                    # Ensure values are float
                    valid_data = pd.to_numeric(valid_data, errors='coerce')

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

        if 'volume' in data.columns:
            if data['volume'].dtype == object:
                data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

            # Перевірка від'ємного об'єму
            negative_volume = data['volume'] < 0
            if negative_volume.any():
                neg_vol_indices = data.index[negative_volume].tolist()
                issues["negative_volume"] = neg_vol_indices
                self.logger.warning(f"Знайдено {len(neg_vol_indices)} записів з від'ємним об'ємом")

            # Перевірка аномального об'єму
            try:
                valid_volume = data['volume'].dropna()
                valid_volume = pd.to_numeric(valid_volume, errors='coerce')

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

    def detect_outliers_ensemble(self, data: pd.DataFrame, methods: List[str] = None,
                                 threshold: float = 3.0, preprocess: bool = True,
                                 fill_method: str = 'mean', contamination: float = 0.1,
                                 min_votes: int = 2) -> Tuple[pd.DataFrame, List]:

        if methods is None:
            methods = ['zscore', 'iqr', 'isolation_forest']

        if len(methods) < min_votes:
            self.logger.warning(f"min_votes ({min_votes}) більше, ніж кількість методів ({len(methods)}). "
                                f"Встановлюємо min_votes = {len(methods)}")
            min_votes = len(methods)

        # Ініціалізуємо результуючий DataFrame для збору голосів
        ensemble_df = pd.DataFrame(index=data.index)
        all_method_results = {}

        # Застосовуємо кожен метод окремо
        for method in methods:
            try:
                outliers_df, _ = self.detect_outliers(
                    data=data,
                    method=method,
                    threshold=threshold,
                    preprocess=preprocess,
                    fill_method=fill_method,
                    contamination=contamination
                )

                if 'is_outlier' in outliers_df.columns:
                    ensemble_df[f'{method}_vote'] = outliers_df['is_outlier']
                    all_method_results[method] = outliers_df
                else:
                    self.logger.warning(f"Метод {method} не виявив жодної аномалії")
                    ensemble_df[f'{method}_vote'] = False

            except Exception as e:
                self.logger.error(f"Помилка при застосуванні методу {method}: {str(e)}")
                ensemble_df[f'{method}_vote'] = False

        # Підрахунок голосів
        vote_columns = [col for col in ensemble_df.columns if col.endswith('_vote')]
        if not vote_columns:
            self.logger.warning("Жоден метод не видав результатів")
            return pd.DataFrame(), []

        ensemble_df['vote_count'] = ensemble_df[vote_columns].sum(axis=1)
        ensemble_df['is_ensemble_outlier'] = ensemble_df['vote_count'] >= min_votes

        # Формування результатів
        ensemble_outlier_indices = data.index[ensemble_df['is_ensemble_outlier']].tolist()

        self.logger.info(f"Ансамблеве виявлення аномалій завершено. "
                         f"Знайдено {len(ensemble_outlier_indices)} аномалій із {min_votes}+ голосами")

        # Додаємо інформацію про індивідуальні методи
        for method in methods:
            if method in all_method_results:
                # Виключаємо колонку is_outlier, щоб уникнути дублювання
                method_cols = [col for col in all_method_results[method].columns if col != 'is_outlier']
                for col in method_cols:
                    ensemble_df[col] = all_method_results[method][col]

        return ensemble_df, ensemble_outlier_indices