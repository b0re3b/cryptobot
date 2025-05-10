from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import decimal
import warnings

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnomalyDetector:

    def __init__(self, logger: Any):
        self.logger = logger

    def ensure_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """Перетворює decimal.Decimal значення на float."""
        result = df.copy()
        for col in result.columns:
            if result[col].dtype == object:
                has_decimal = any(isinstance(x, decimal.Decimal) for x in result[col].dropna())
                if has_decimal:
                    self.logger.info(f"Converting decimal.Decimal values to float in column {col}")
                    result[col] = result[col].apply(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)
        return result

    def _preprocess_data(self, data: pd.DataFrame, numeric_cols: List[str],
                         fill_method: str = 'median') -> pd.DataFrame:
        """Попередня обробка даних з врахуванням специфіки криптовалют."""
        if data.empty or not numeric_cols:
            self.logger.warning("Порожні вхідні дані або відсутні числові колонки")
            return data

        processed_data = self.ensure_float(data)

        # 1. Заповнення відсутніх значень - краще використовувати медіану для криптовалют через викиди
        for col in numeric_cols:
            if col not in processed_data.columns:
                continue

            # Переконуємося, що колонка є числовою
            if processed_data[col].dtype == object:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

            if processed_data[col].isna().any():
                if fill_method == 'mean':
                    fill_value = processed_data[col].mean()
                elif fill_method == 'median':  # Рекомендований для крипти
                    fill_value = processed_data[col].median()
                elif fill_method == 'ffill':
                    processed_data[col] = processed_data[col].ffill()
                    continue
                elif fill_method == 'bfill':
                    processed_data[col] = processed_data[col].bfill()
                    continue
                else:
                    fill_value = 0
                processed_data[col] = processed_data[col].fillna(fill_value)

        # 2. Логарифмічне перетворення для криптовалютних даних з високою скошеністю
        for col in numeric_cols:
            if processed_data[col].dtype == object:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

            # Пропускаємо колонки з одним унікальним значенням
            unique_values = processed_data[col].nunique()
            if unique_values <= 1:
                continue

            # Перевірка на викиди та позитивні значення
            if (processed_data[col] > 0).all():
                skewness = processed_data[col].skew()
                # Криптовалютні дані часто сильно скошені
                if abs(skewness) > 1.5:  # Підвищуємо поріг для крипто
                    processed_data[f'{col}_log'] = np.log1p(processed_data[col])
                    self.logger.info(f"Застосовано логарифмічне перетворення для {col} (скошеність={skewness:.2f})")

        # 3. Видалення дублікатів індексу
        if isinstance(processed_data.index, pd.DatetimeIndex):
            duplicates = processed_data.index.duplicated()
            if duplicates.any():
                processed_data = processed_data[~duplicates]
                self.logger.info(f"Видалено {duplicates.sum()} дублікатів індексу")

        # 4. Сортування за індексом (для часових рядів криптовалют)
        if isinstance(processed_data.index, pd.DatetimeIndex):
            processed_data = processed_data.sort_index()

        return processed_data

    def detect_outliers(self, data: pd.DataFrame, method: str = 'crypto_specific',
                        threshold: float = 5.0, preprocess: bool = True,
                        fill_method: str = 'median', contamination: float = 0.05) -> Tuple[pd.DataFrame, List]:
        """Виявлення аномалій з використанням різних методів, оптимізовано для криптовалют."""
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для виявлення аномалій")
            return pd.DataFrame(), []

        # Валідація параметрів
        if method not in ['zscore', 'iqr', 'isolation_forest', 'crypto_specific', 'ensemble']:
            self.logger.warning(f"Непідтримуваний метод: {method}. Використовуємо 'crypto_specific'")
            method = 'crypto_specific'

        # Збільшуємо стандартний поріг для крипто - вони мають більше природних коливань
        if threshold <= 0:
            self.logger.warning(f"Отримано недопустиме порогове значення: {threshold}. Встановлено значення 5")
            threshold = 5.0

        # Зменшуємо contamination для Isolation Forest для криптовалют - менше помилкових спрацьовувань
        if contamination <= 0 or contamination > 0.5:
            self.logger.warning(f"Неправильне значення contamination: {contamination}. Використовуємо 0.05")
            contamination = 0.05

        self.logger.info(f"Початок виявлення аномалій методом {method} з порогом {threshold}")

        data = self.ensure_float(data)

        # Вибір числових колонок
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Перетворення об'єктів на числа, коли можливо
        for col in data.columns:
            if col not in numeric_cols and data[col].dtype == object:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    if not data[col].isna().all():
                        numeric_cols.append(col)
                except:
                    pass

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

        # Результати зберігаємо в словнику, без перейменування колонок
        anomalies = {}
        all_outlier_indices = set()

        try:
            if method == 'ensemble':
                # Використовуємо комбінацію методів для криптовалют
                self.logger.info("Використовуємо ансамблевий метод виявлення аномалій")
                self._detect_crypto_specific_anomalies(processed_data, threshold, anomalies, all_outlier_indices)
                self._detect_robust_zscore_anomalies(processed_data, numeric_cols, threshold, anomalies,
                                                     all_outlier_indices)
                self._detect_robust_iqr_anomalies(processed_data, numeric_cols, threshold * 0.8, anomalies,
                                                  all_outlier_indices)

                if SKLEARN_AVAILABLE and len(processed_data) >= 50:  # Мінімум 50 точок для надійності
                    self._detect_isolation_forest_anomalies(processed_data, numeric_cols, contamination, anomalies,
                                                            all_outlier_indices)

            elif method == 'zscore':
                self._detect_robust_zscore_anomalies(processed_data, numeric_cols, threshold, anomalies,
                                                     all_outlier_indices)

            elif method == 'iqr':
                self._detect_robust_iqr_anomalies(processed_data, numeric_cols, threshold, anomalies,
                                                  all_outlier_indices)

            elif method == 'isolation_forest':
                if not SKLEARN_AVAILABLE:
                    self.logger.error("Для використання методу 'isolation_forest' необхідно встановити scikit-learn")
                    return pd.DataFrame(), []
                self._detect_isolation_forest_anomalies(processed_data, numeric_cols, contamination, anomalies,
                                                        all_outlier_indices)

            elif method == 'crypto_specific':
                self._detect_crypto_specific_anomalies(processed_data, threshold, anomalies, all_outlier_indices)

            else:
                self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
                return pd.DataFrame(), []

        except Exception as e:
            self.logger.error(f"Помилка під час виявлення аномалій методом {method}: {str(e)}")
            return pd.DataFrame(), []

        # Перетворюємо результати на DataFrame
        result_df = pd.DataFrame(index=data.index)

        # Додаємо інформацію про аномалії без перейменування колонок
        for anomaly_type, indices in anomalies.items():
            if indices:
                result_df[anomaly_type] = False
                result_df.loc[indices, anomaly_type] = True

        # Додаємо загальний індикатор аномалій
        if not result_df.empty:
            result_df['is_anomaly'] = result_df.any(axis=1)
        else:
            result_df['is_anomaly'] = False

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return result_df, outlier_indices

    def _detect_robust_zscore_anomalies(self, data: pd.DataFrame, numeric_cols: List[str],
                                        threshold: float, anomalies: Dict[str, List],
                                        all_outlier_indices: set) -> None:
        """Виявлення аномалій за допомогою робастного Z-Score."""
        # Відфільтруємо колонки з проблемами
        valid_cols = []

        for col in numeric_cols:
            if col not in data.columns:
                continue

            # Переконуємося, що дані числові та відкидаємо NaN
            valid_data = pd.to_numeric(data[col], errors='coerce').dropna()

            if valid_data.empty or valid_data.nunique() <= 1:
                continue

            median = valid_data.median()
            # MAD - робастна альтернатива std для криптовалют
            mad = np.median(np.abs(valid_data - median)) * 1.4826

            # Запобігаємо діленню на нуль
            if np.isclose(mad, 0) or pd.isna(mad):
                continue

            valid_cols.append(col)

            # Ініціалізуємо серію для робастного Z-Score
            z_scores = pd.Series(np.nan, index=data.index)
            valid_indices = valid_data.index
            z_scores.loc[valid_indices] = np.abs((valid_data - median) / mad)

            # Визначаємо аномалії з більшим порогом для криптовалют
            outliers = z_scores > threshold
            outliers = outliers.fillna(False)

            if outliers.any():
                # Зберігаємо індекси аномалій без перейменування колонок
                anomaly_indices = data.index[outliers].tolist()
                anomalies[f"zscore_anomaly_{col}"] = anomaly_indices
                outlier_count = outliers.sum()
                self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (zscore)")
                all_outlier_indices.update(anomaly_indices)

    def _detect_robust_iqr_anomalies(self, data: pd.DataFrame, numeric_cols: List[str],
                                     threshold: float, anomalies: Dict[str, List],
                                     all_outlier_indices: set) -> None:
        """Виявлення аномалій за методом IQR з адаптацією для криптовалют."""
        for col in numeric_cols:
            if col not in data.columns:
                continue

            # Переконуємося, що дані числові
            valid_data = pd.to_numeric(data[col], errors='coerce').dropna()

            if len(valid_data) < 4 or valid_data.nunique() <= 1:
                continue

            try:
                # Для криптовалют використовуємо розширені квантилі
                Q1 = valid_data.quantile(0.10)  # Розширені квантилі для криптовалют
                Q3 = valid_data.quantile(0.90)

                # Вирішення проблеми Q1=Q3 для криптовалют
                if np.isclose(Q1, Q3):
                    # Використовуємо медіану і мінімальне відхилення
                    median = valid_data.median()
                    abs_deviations = np.abs(valid_data - median)
                    non_zero_deviations = abs_deviations[abs_deviations > 0]

                    if len(non_zero_deviations) > 0:
                        min_non_zero = non_zero_deviations.min()
                        # Використовуємо більший множник для криптовалют
                        artificial_range = min_non_zero * 8

                        lower_bound = median - threshold * artificial_range
                        upper_bound = median + threshold * artificial_range
                    else:
                        continue
                else:
                    # Розширений IQR для криптовалют
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                # Створюємо серію з індексами original data
                outliers = pd.Series(False, index=data.index)
                valid_indices = valid_data.index
                outliers.loc[valid_indices] = (valid_data < lower_bound) | (valid_data > upper_bound)

                if outliers.any():
                    # Зберігаємо індекси аномалій без перейменування колонок
                    anomaly_indices = data.index[outliers].tolist()
                    anomalies[f"iqr_anomaly_{col}"] = anomaly_indices
                    outlier_count = outliers.sum()
                    self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (IQR)")
                    all_outlier_indices.update(anomaly_indices)

            except Exception as e:
                self.logger.error(f"Помилка при застосуванні IQR методу до колонки {col}: {str(e)}")

    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame, numeric_cols: List[str],
                                           contamination: float, anomalies: Dict[str, List],
                                           all_outlier_indices: set) -> None:
        """Виявлення аномалій за допомогою Isolation Forest з оптимізацією для криптовалют."""
        if not SKLEARN_AVAILABLE:
            self.logger.error("Для використання методу 'isolation_forest' необхідно встановити scikit-learn")
            return

        # Перевірка наявності достатньої кількості даних
        if len(data) < 20:  # Для крипто рекомендується більше даних
            self.logger.warning("Недостатньо даних для Isolation Forest (рекомендовано мінімум 20 записів)")
            return

        # Підготовка даних - виділення тільки числових колонок без NaN
        numeric_cols = [col for col in numeric_cols if col in data.columns]
        if not numeric_cols:
            return

        # Створюємо копію щоб уникнути попереджень SettingWithCopyWarning
        X = data[numeric_cols].copy()

        # Підготовка даних для криптовалют
        non_constant_cols = []
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            # Перевірка варіативності
            if X[col].nunique() > 1:
                non_constant_cols.append(col)

        if not non_constant_cols:
            self.logger.warning("Всі колонки мають однакові значення, Isolation Forest буде неефективним")
            return

        X = X[non_constant_cols]

        # Заповнення пропущених значень медіаною (краще для скошених даних криптовалют)
        for col in non_constant_cols:
            if X[col].isna().any():
                col_median = X[col].median()
                if pd.isna(col_median):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(col_median)

        # Перевірка на наявність NaN після заповнення
        if X.isna().any().any():
            X = X.fillna(0)

        # Оптимізовані параметри для криптовалют
        n_estimators = 300  # Більше дерев для кращої робусності
        max_samples = min(256, int(len(X) * 0.8))  # Обмежуємо розмір підвибірки

        try:
            # Параметри оптимізовані для криптовалютних даних
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=n_estimators,
                max_samples=max_samples,
                bootstrap=True  # Використання бутстрепу для більш робустних результатів
            )

            predictions = model.fit_predict(X)

            # -1 для аномалій, 1 для нормальних значень
            anomaly_mask = predictions == -1
            if np.any(anomaly_mask):
                anomaly_indices = data.index[anomaly_mask].tolist()
                anomalies["isolation_forest_anomaly"] = anomaly_indices
                anomaly_count = len(anomaly_indices)
                self.logger.info(f"Знайдено {anomaly_count} аномалій методом Isolation Forest")
                all_outlier_indices.update(anomaly_indices)

        except Exception as e:
            self.logger.error(f"Помилка при використанні Isolation Forest: {str(e)}")

    def _detect_crypto_specific_anomalies(self, data: pd.DataFrame, threshold: float,
                                          anomalies: Dict[str, List], all_outlier_indices: set) -> None:
        """Спеціальний метод виявлення аномалій для криптовалютних даних."""
        # Перевірка наявності типових колонок для криптовалютних даних
        crypto_price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        volume_col = 'volume' if 'volume' in data.columns else None

        if not crypto_price_cols and not volume_col:
            self.logger.warning("Не знайдено типових колонок для криптовалютних даних")
            return

        # 1. Виявлення різких стрибків цін - специфічно для криптовалют використовуємо більші пороги
        if len(crypto_price_cols) > 0:
            for col in crypto_price_cols:
                if data[col].dtype == object:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

                # Якщо дані сортовані за часом, перевіряємо стрибки
                if isinstance(data.index, pd.DatetimeIndex) and data.index.is_monotonic_increasing:
                    pct_change = data[col].pct_change().fillna(0)

                    # Для криптовалют використовуємо більший поріг - 15% (замість 10%)
                    large_pos_jumps = pct_change > 0.15  # 15% стрибок вгору
                    large_neg_jumps = pct_change < -0.15  # 15% стрибок вниз

                    price_jumps = large_pos_jumps | large_neg_jumps

                    if price_jumps.any():
                        jump_indices = data.index[price_jumps].tolist()
                        anomalies[f"price_jump_{col}"] = jump_indices
                        self.logger.info(f"Знайдено {price_jumps.sum()} різких змін ціни у колонці {col}")
                        all_outlier_indices.update(jump_indices)

                # Перевірка на нульові ціни (неможливо для більшості криптовалют)
                invalid_prices = data[col] <= 0
                if invalid_prices.any():
                    invalid_indices = data.index[invalid_prices].tolist()
                    anomalies[f"invalid_price_{col}"] = invalid_indices
                    self.logger.info(f"Знайдено {invalid_prices.sum()} нульових або від'ємних значень у колонці {col}")
                    all_outlier_indices.update(invalid_indices)

        # 2. Перевірка узгодженості OHLC даних
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Перетворення на числові дані
            for col in ['open', 'high', 'low', 'close']:
                if data[col].dtype == object:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            # High має бути >= всіх інших цін
            invalid_high = (data['high'] < data['open']) | (data['high'] < data['low']) | (data['high'] < data['close'])

            # Low має бути <= всіх інших цін
            invalid_low = (data['low'] > data['open']) | (data['low'] > data['high']) | (data['low'] > data['close'])

            invalid_ohlc = invalid_high | invalid_low

            if invalid_ohlc.any():
                invalid_indices = data.index[invalid_ohlc].tolist()
                anomalies["invalid_ohlc"] = invalid_indices
                self.logger.info(f"Знайдено {invalid_ohlc.sum()} записів з неузгодженими OHLC даними")
                all_outlier_indices.update(invalid_indices)

        # 3. Аномалії об'єму торгів (дуже важливо для криптовалют)
        if volume_col:
            if data[volume_col].dtype == object:
                data[volume_col] = pd.to_numeric(data[volume_col], errors='coerce')

            # Перевірка на нульовий об'єм (підозріло для активно торгованих криптовалют)
            zero_volume = data[volume_col] == 0
            if zero_volume.any():
                zero_vol_indices = data.index[zero_volume].tolist()
                anomalies["zero_volume"] = zero_vol_indices
                # Не завжди додаємо до загальних аномалій, бо нульовий об'єм може бути нормальним
                self.logger.info(f"Знайдено {zero_volume.sum()} записів з нульовим об'ємом")

            # Перевірка на аномально високий об'єм - криптовалюти можуть мати великі стрибки об'єму
            rolling_vol = data[volume_col].rolling(window=24, min_periods=1).median()

            with np.errstate(divide='ignore', invalid='ignore'):
                vol_ratio = data[volume_col] / rolling_vol
                vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Для криптовалют підвищуємо поріг до 20x
            high_volume = vol_ratio > 20  # Об'єм у 20+ разів вище медіанного

            if high_volume.any():
                high_vol_indices = data.index[high_volume].tolist()
                anomalies["high_volume"] = high_vol_indices
                self.logger.info(f"Знайдено {high_volume.sum()} записів з аномально високим об'ємом")
                all_outlier_indices.update(high_vol_indices)

            # Від'ємний об'єм - неможливо для криптовалют
            negative_volume = data[volume_col] < 0
            if negative_volume.any():
                neg_vol_indices = data.index[negative_volume].tolist()
                anomalies["negative_volume"] = neg_vol_indices
                self.logger.info(f"Знайдено {negative_volume.sum()} записів з від'ємним об'ємом")
                all_outlier_indices.update(neg_vol_indices)

        # 4. Виявлення Flash Crash та Flash Pump (специфічно для крипторинків)
        if all(col in data.columns for col in ['high', 'low']) and len(data) > 10:
            try:
                # Обчислюємо амплітуду коливань (high-low)/(0.5*(high+low))
                amplitude = (data['high'] - data['low']) / (0.5 * (data['high'] + data['low']))
                # Знаходимо медіанну амплітуду
                median_amplitude = amplitude.median()

                if median_amplitude > 0:
                    # Flash Crash/Pump - коли амплітуда в N разів перевищує медіанну
                    flash_events = amplitude > (median_amplitude * 10)  # Для криптовалют використовуємо множник 10

                    if flash_events.any():
                        flash_indices = data.index[flash_events].tolist()
                        anomalies["flash_event"] = flash_indices
                        self.logger.info(f"Знайдено {flash_events.sum()} різких цінових сплесків (Flash events)")
                        all_outlier_indices.update(flash_indices)
            except Exception as e:
                self.logger.error(f"Помилка при виявленні Flash events: {str(e)}")

        # 5. Виявлення помилкових "flat" періодів (однакова ціна протягом тривалого часу)
        if 'close' in data.columns and len(data) > 20:
            try:
                # Для активних крипторинків ціна не повинна стояти на місці довго
                rolling_std = data['close'].rolling(window=12, min_periods=3).std()
                flat_periods = rolling_std == 0

                # Ігноруємо короткі flat періоди (менше 3 послідовних)
                if flat_periods.any():
                    # Використовуємо підхід для виявлення послідовних груп
                    flat_groups = (flat_periods != flat_periods.shift()).cumsum()
                    flat_counts = flat_periods.groupby(flat_groups).transform('sum')

                    # Враховуємо тільки тривалі flat періоди (3+ послідовних точок)
                    significant_flat = (flat_periods & (flat_counts >= 3))

                    if significant_flat.any():
                        flat_indices = data.index[significant_flat].tolist()
                        anomalies["flat_price_period"] = flat_indices
                        self.logger.info(f"Знайдено {significant_flat.sum()} точок у підозрілих flat-price періодах")
                        all_outlier_indices.update(flat_indices)
            except Exception as e:
                self.logger.error(f"Помилка при виявленні flat періодів: {str(e)}")

    def validate_datetime_index(self, data: pd.DataFrame, issues: Dict[str, Any]) -> None:
        """Перевірка часового індексу на проблеми."""
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

    def validate_price_data(self, data: pd.DataFrame, price_jump_threshold: float,
                            issues: Dict[str, Any]) -> None:
        """Валідація цінових даних з урахуванням специфіки Binance (всі значення додатні)."""
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]

        # Переконуємося, що всі цінові колонки числові
        for col in price_cols:
            if data[col].dtype == object:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        if len(price_cols) == 4:
            # Перевірка на некоректні співвідношення high і low
            invalid_hl = data['high'] < data['low']
            if invalid_hl.any():
                invalid_hl_indices = data.index[invalid_hl].tolist()
                issues["invalid_high_low"] = invalid_hl_indices
                self.logger.warning(f"Знайдено {len(invalid_hl_indices)} записів де high < low")

            # Перевірка на неконсистентність OHLC
            invalid_ohlc = (
                    (data['high'] < data['open']) |
                    (data['high'] < data['close']) |
                    (data['low'] > data['open']) |
                    (data['low'] > data['close'])
            )
            if invalid_ohlc.any():
                invalid_ohlc_indices = data.index[invalid_ohlc].tolist()
                issues["inconsistent_ohlc"] = invalid_ohlc_indices
                self.logger.warning(f"Знайдено {len(invalid_ohlc_indices)} записів з неконсистентними OHLC даними")

            # Для даних Binance - перевірка на нульові ціни (замість від'ємних)
            for col in price_cols:
                zero_prices = data[col] == 0
                if zero_prices.any():
                    zero_price_indices = data.index[zero_prices].tolist()
                    issues[f"zero_{col}"] = zero_price_indices
                    self.logger.warning(f"Знайдено {len(zero_price_indices)} записів з нульовими значеннями у {col}")

            # Перевірка різких стрибків цін
            for col in price_cols:
                try:
                    # Безпечне обчислення відсоткової зміни з обробкою NaN
                    valid_data = data[col].dropna()
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

    def validate_volume_data(self, data: pd.DataFrame, volume_anomaly_threshold: float,
                             issues: Dict[str, Any]) -> None:
        """Валідація даних об'єму з урахуванням специфіки Binance (всі значення додатні)."""
        if 'volume' in data.columns:
            if data['volume'].dtype == object:
                data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

            # Для даних Binance - перевірка на нульовий об'єм замість від'ємного
            zero_volume = data['volume'] == 0
            if zero_volume.any():
                zero_vol_indices = data.index[zero_volume].tolist()
                issues["zero_volume"] = zero_vol_indices
                self.logger.warning(f"Знайдено {len(zero_vol_indices)} записів з нульовим об'ємом")

            # Перевірка аномального об'єму
            try:
                valid_volume = data['volume'].dropna()
                valid_volume = pd.to_numeric(valid_volume, errors='coerce')

                if not valid_volume.empty:
                    # Використовуємо робастний підхід для криптовалют
                    volume_median = valid_volume.median()
                    volume_mad = (valid_volume - volume_median).abs().median() * 1.4826  # масштабований MAD

                    if volume_mad > 0:  # Уникаємо ділення на нуль
                        volume_zscore = np.abs((valid_volume - volume_median) / volume_mad)
                        volume_anomalies = volume_zscore > volume_anomaly_threshold

                        if volume_anomalies.any():
                            vol_anomaly_indices = valid_volume.index[volume_anomalies].tolist()
                            issues["volume_anomalies"] = vol_anomaly_indices
                            self.logger.warning(f"Знайдено {len(vol_anomaly_indices)} записів з аномальним об'ємом")
                    else:
                        # Альтернативний метод при близьких до однорідних даних
                        if volume_median > 0:
                            # Перевіряємо на стрибки більше ніж у 5 разів від медіани
                            volume_anomalies = valid_volume > (volume_median * 5)
                            if volume_anomalies.any():
                                vol_anomaly_indices = valid_volume.index[volume_anomalies].tolist()
                                issues["volume_spikes"] = vol_anomaly_indices
                                self.logger.warning(f"Знайдено {len(vol_anomaly_indices)} аномальних стрибків об'єму")
            except Exception as e:
                self.logger.error(f"Помилка при аналізі аномалій об'єму: {str(e)}")

    def validate_data_values(self, data: pd.DataFrame, issues: Dict[str, Any]) -> None:
        """Загальна валідація даних на відсутні та некоректні значення."""
        # Перевірка на NaN значення
        na_counts = data.isna().sum()
        cols_with_na = na_counts[na_counts > 0].to_dict()
        if cols_with_na:
            issues["na_counts"] = cols_with_na
            self.logger.warning(f"Знайдено відсутні значення у колонках: {list(cols_with_na.keys())}")

        # Перевірка на нескінченні значення
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                inf_counts = {col: np.isinf(data[col]).sum() for col in numeric_cols}
                cols_with_inf = {col: count for col, count in inf_counts.items() if count > 0}

                if cols_with_inf:
                    issues["inf_counts"] = cols_with_inf
                    self.logger.warning(f"Знайдено нескінченні значення у колонках: {list(cols_with_inf.keys())}")
        except Exception as e:
            self.logger.error(f"Помилка при перевірці нескінченних значень: {str(e)}")

    def detect_outliers_essemble(self, data: pd.DataFrame, method: str = 'zscore',
                        threshold: float = 3.0, preprocess: bool = True,
                        fill_method: str = 'mean', contamination: float = 0.1) -> Tuple[pd.DataFrame, List]:
        """Виявлення аномалій з використанням різних методів."""
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для виявлення аномалій")
            return pd.DataFrame(), []

        # Валідація параметрів
        if method not in ['zscore', 'iqr', 'isolation_forest', 'crypto_specific']:
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

        data = self.ensure_float(data)

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

        # Використовуємо словник для зберігання аномалій замість додавання колонок з суфіксом _outlier
        anomalies_dict = {}
        all_outlier_indices = set()

        try:
            if method == 'zscore':
                self._detect_robust_zscore_anomalies(processed_data, numeric_cols, threshold, anomalies_dict,
                                            all_outlier_indices)
            elif method == 'iqr':
                self._detect_robust_iqr_anomalies(processed_data, numeric_cols, threshold, anomalies_dict, all_outlier_indices)
            elif method == 'isolation_forest':
                # Для isolation_forest використовуємо contamination замість threshold
                self._detect_isolation_forest_anomalies(processed_data, numeric_cols, contamination, anomalies_dict,
                                                      all_outlier_indices)
            elif method == 'crypto_specific':
                # Метод, специфічний для криптовалют
                self._detect_crypto_specific_anomalies(processed_data, numeric_cols, threshold, anomalies_dict,
                                                     all_outlier_indices)
            else:
                self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
                return pd.DataFrame(), []

        except Exception as e:
            self.logger.error(f"Помилка під час виявлення аномалій методом {method}: {str(e)}")
            return pd.DataFrame(), []

        # Створюємо результуючий DataFrame з аномаліями
        outliers_df = pd.DataFrame(index=data.index)

        # Додаємо виявлені аномалії до результуючого DataFrame
        for col_name, anomalies in anomalies_dict.items():
            outliers_df[col_name] = pd.Series(False, index=data.index)
            outliers_df.loc[anomalies, col_name] = True

        # Додаємо загальну колонку is_outlier
        if not outliers_df.empty:
            outliers_df['is_outlier'] = outliers_df.any(axis=1)

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return outliers_df, outlier_indices