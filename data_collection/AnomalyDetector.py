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

    def _preprocess_data(self, data: pd.DataFrame, numeric_cols: List[str], fill_method: str = 'mean') -> pd.DataFrame:
        """Попередня обробка даних з врахуванням специфіки криптовалют."""
        if data.empty or not numeric_cols:
            self.logger.warning("Порожні вхідні дані або відсутні числові колонки")
            return data

        processed_data = self.ensure_float(data)

        # 1. Заповнення відсутніх значень
        for col in numeric_cols:
            if col not in processed_data.columns:
                continue

            # Переконуємося, що колонка є числовою
            if processed_data[col].dtype == object:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

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
                    fill_value = 0
                processed_data[col] = processed_data[col].fillna(fill_value)

        # 2. Криптовалютні дані можуть мати екстремальні значення - використаємо робастну нормалізацію
        for col in numeric_cols:
            # Перевіряємо, чи всі значення однакові
            if processed_data[col].dtype == object:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

            unique_values = processed_data[col].nunique()

            if unique_values <= 1:
                self.logger.warning(
                    f"Колонка {col} має {unique_values} унікальних значень. Пропускаємо логарифмічне перетворення.")
                continue

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
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Перетворення об'єктів на числа, коли можливо
        for col in data.columns:
            if col not in numeric_cols and data[col].dtype == object:
                try:
                    # Якщо колонка містить числа як строки
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    if not data[col].isna().all():  # Якщо не всі значення стали NaN
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

        outliers_df = pd.DataFrame(index=data.index)
        all_outlier_indices = set()

        try:
            if method == 'zscore':
                self.detect_zscore_outliers(processed_data, numeric_cols, threshold, outliers_df, all_outlier_indices)
            elif method == 'iqr':
                self.detect_iqr_outliers(processed_data, numeric_cols, threshold, outliers_df, all_outlier_indices)
            elif method == 'isolation_forest':
                # Для isolation_forest використовуємо contamination замість threshold
                self.detect_isolation_forest_outliers(processed_data, numeric_cols, contamination, outliers_df,
                                                      all_outlier_indices)
            elif method == 'crypto_specific':
                # Новий метод, специфічний для криптовалют
                self.detect_crypto_specific_outliers(processed_data, numeric_cols, threshold, outliers_df,
                                                     all_outlier_indices)
            else:
                self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
                return pd.DataFrame(), []

        except Exception as e:
            self.logger.error(f"Помилка під час виявлення аномалій методом {method}: {str(e)}")
            return pd.DataFrame(), []

        if not outliers_df.empty:
            # Перевіряємо, чи були знайдені аномалії в будь-якій колонці
            cols_to_check = [col for col in outliers_df.columns if col.endswith('_outlier')]
            if cols_to_check:
                outliers_df['is_outlier'] = outliers_df[cols_to_check].any(axis=1)
            else:
                outliers_df['is_outlier'] = False

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return outliers_df, outlier_indices

    def detect_zscore_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                               threshold: float, outliers_df: pd.DataFrame,
                               all_outlier_indices: set) -> None:
        """Виявлення аномалій за методом Z-Score."""
        # Відфільтруємо колонки з проблемами
        valid_cols = []

        for col in numeric_cols:
            if col not in data.columns:
                continue

            if data[col].isna().all():
                self.logger.warning(f"Колонка {col} містить лише NaN значення, пропускаємо")
                continue

            # Переконуємося, що дані числові
            valid_data = pd.to_numeric(data[col], errors='coerce').dropna()

            if valid_data.empty:
                self.logger.warning(f"Колонка {col} не містить валідних числових даних")
                continue

            # Перевірка на однорідність даних
            if valid_data.nunique() <= 1:
                self.logger.warning(
                    f"Колонка {col} має {valid_data.nunique()} унікальних значень, Z-Score буде неефективним")
                continue

            std = valid_data.std()
            if np.isclose(std, 0) or pd.isna(std):
                self.logger.warning(f"Колонка {col} має нульове стандартне відхилення або NaN")
                continue

            valid_cols.append(col)

        if not valid_cols:
            self.logger.warning("Немає валідних колонок для Z-Score аналізу")
            return

        # Обчислимо Z-Score для всіх валідних колонок
        for col in valid_cols:
            valid_data = pd.to_numeric(data[col], errors='coerce').dropna()

            # Використовуємо робастні оцінки для криптовалютних даних
            median = valid_data.median()
            # MAD (Median Absolute Deviation) - робастна альтернатива std
            mad = np.median(np.abs(valid_data - median)) * 1.4826  # множник для приведення MAD до масштабу std

            # Запобігаємо діленню на нуль
            if np.isclose(mad, 0):
                self.logger.warning(f"MAD для колонки {col} дорівнює нулю, використовуємо стандартний Z-Score")
                mean = valid_data.mean()
                std = valid_data.std()
                if np.isclose(std, 0):
                    self.logger.warning(f"Std для колонки {col} також дорівнює нулю, пропускаємо")
                    continue

                # Ініціалізуємо серію для Z-Score з NaN
                z_scores = pd.Series(np.nan, index=data.index)

                # Заповнюємо тільки валідні індекси
                valid_indices = valid_data.index
                z_scores.loc[valid_indices] = np.abs((valid_data - mean) / std)
            else:
                # Ініціалізуємо серію для робастного Z-Score з NaN
                z_scores = pd.Series(np.nan, index=data.index)

                # Заповнюємо тільки валідні індекси
                valid_indices = valid_data.index
                z_scores.loc[valid_indices] = np.abs((valid_data - median) / mad)

            # Визначаємо аномалії
            outliers = z_scores > threshold

            # Заповнюємо False для NaN значень
            outliers = outliers.fillna(False)

            outliers_df[f'{col}_outlier'] = outliers

            if outliers.any():
                outlier_count = outliers.sum()
                self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (zscore)")
                all_outlier_indices.update(data.index[outliers])

    def detect_iqr_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                            threshold: float, outliers_df: pd.DataFrame,
                            all_outlier_indices: set) -> None:
        """Виявлення аномалій за методом IQR з поліпшеною обробкою Q1=Q3."""
        for col in numeric_cols:
            if col not in data.columns:
                continue

            # Переконуємося, що дані числові
            valid_data = pd.to_numeric(data[col], errors='coerce').dropna()

            if len(valid_data) < 4:
                self.logger.warning(f"Недостатньо даних у колонці {col} для IQR методу")
                continue

            # Перевірка на однорідність даних
            unique_count = valid_data.nunique()
            if unique_count <= 1:
                self.logger.warning(f"Колонка {col} має {unique_count} унікальних значень, IQR метод буде неефективним")
                continue

            try:
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)

                # Вирішення проблеми Q1=Q3 для криптовалют
                if np.isclose(Q1, Q3):
                    self.logger.warning(f"Q1=Q3 для колонки {col}. Застосовуємо модифікований метод.")

                    # Спробуємо використати більш широкий інтервал квантилів
                    Q1_alt = valid_data.quantile(0.10)
                    Q3_alt = valid_data.quantile(0.90)

                    if np.isclose(Q1_alt, Q3_alt):
                        # Якщо все ще однакові, спробуємо взяти відхилення від медіани
                        median = valid_data.median()

                        # Знаходимо ненульові відхилення від медіани
                        abs_deviations = np.abs(valid_data - median)
                        non_zero_deviations = abs_deviations[abs_deviations > 0]

                        if len(non_zero_deviations) > 0:
                            min_non_zero = non_zero_deviations.min()
                            # Використовуємо мінімальне ненульове відхилення як штучний IQR
                            artificial_iqr = min_non_zero

                            lower_bound = median - threshold * artificial_iqr * 3
                            upper_bound = median + threshold * artificial_iqr * 3

                            self.logger.info(f"Використано штучний IQR для колонки {col}")
                        else:
                            self.logger.warning(
                                f"Неможливо застосувати модифікований IQR для колонки {col}. Всі значення рівні {median}.")
                            continue
                    else:
                        # Використовуємо альтернативні квантилі
                        IQR = Q3_alt - Q1_alt
                        lower_bound = Q1_alt - threshold * IQR
                        upper_bound = Q3_alt + threshold * IQR
                        self.logger.info(f"Використано альтернативні квантилі (10% і 90%) для колонки {col}")
                else:
                    # Стандартний IQR метод
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                # Створюємо серію з індексами original data
                outliers = pd.Series(False, index=data.index)

                # Заповнюємо тільки для валідних індексів
                valid_indices = valid_data.index
                outliers.loc[valid_indices] = (valid_data < lower_bound) | (valid_data > upper_bound)

                outliers_df[f'{col}_outlier'] = outliers

                if outliers.any():
                    outlier_count = outliers.sum()
                    self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (IQR)")
                    all_outlier_indices.update(data.index[outliers])

            except Exception as e:
                self.logger.error(f"Помилка при застосуванні IQR методу до колонки {col}: {str(e)}")

    def detect_isolation_forest_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                                         contamination: float, outliers_df: pd.DataFrame,
                                         all_outlier_indices: set) -> None:
        """Виявлення аномалій за допомогою Isolation Forest."""
        if not SKLEARN_AVAILABLE:
            self.logger.error("Для використання методу 'isolation_forest' необхідно встановити scikit-learn")
            return

        # Перевірка наявності достатньої кількості даних
        if len(data) < 10:
            self.logger.warning("Недостатньо даних для Isolation Forest (потрібно мінімум 10 записів)")
            return

        # Підготовка даних - виділення тільки числових колонок без NaN
        numeric_cols = [col for col in numeric_cols if col in data.columns]
        if not numeric_cols:
            self.logger.warning("Відсутні числові колонки для Isolation Forest")
            return

        # Створюємо копію щоб уникнути попереджень SettingWithCopyWarning
        X = data[numeric_cols].copy()

        # Перевірка на однорідність даних
        non_constant_cols = []
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            if X[col].nunique() > 1:
                non_constant_cols.append(col)

        if not non_constant_cols:
            self.logger.warning("Всі колонки мають однакові значення, Isolation Forest буде неефективним")
            return

        X = X[non_constant_cols]

        # Заповнення пропущених значень
        for col in non_constant_cols:
            if X[col].isna().any():
                col_median = X[col].median()
                if pd.isna(col_median):  # Якщо медіана також NaN
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(col_median)

        # Перевірка на наявність NaN після заповнення
        if X.isna().any().any():
            self.logger.warning("Залишились NaN після заповнення. Вони будуть замінені на 0")
            X = X.fillna(0)

        # Валідація параметра contamination
        if contamination <= 0 or contamination > 0.5:
            self.logger.warning(f"Неправильне значення contamination: {contamination}. Використовуємо 0.1")
            contamination = 0.1

        # Для криптовалют часто ефективніше використовувати більшу кількість дерев
        n_estimators = 200  # Збільшено для кращої робусності

        try:
            # Оптимізація параметрів для криптовалютних даних
            model = IsolationForest(contamination=contamination,
                                    random_state=42,
                                    n_estimators=n_estimators,
                                    max_samples='auto',
                                    bootstrap=True)  # Використання бутстрепу для більш робустних результатів

            predictions = model.fit_predict(X)

            # -1 для викидів, 1 для нормальних значень
            outliers = pd.Series(predictions == -1, index=data.index)
            outliers_df['isolation_forest_outlier'] = outliers

            if outliers.any():
                outlier_indices = data.index[outliers]
                outlier_count = len(outlier_indices)
                self.logger.info(f"Знайдено {outlier_count} аномалій методом Isolation Forest")
                all_outlier_indices.update(outlier_indices)

        except Exception as e:
            self.logger.error(f"Помилка при використанні Isolation Forest: {str(e)}")

    def detect_crypto_specific_outliers(self, data: pd.DataFrame, numeric_cols: List[str],
                                        threshold: float, outliers_df: pd.DataFrame,
                                        all_outlier_indices: set) -> None:
        """Спеціальний метод виявлення аномалій для криптовалютних даних."""
        # Перевірка наявності типових колонок для криптовалютних даних
        crypto_price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        volume_col = 'volume' if 'volume' in data.columns else None

        if not crypto_price_cols and not volume_col:
            self.logger.warning("Не знайдено типових колонок для криптовалютних даних")
            return

        # 1. Виявлення різких стрибків цін (common for crypto)
        if len(crypto_price_cols) > 0:
            for col in crypto_price_cols:
                if data[col].dtype == object:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

                # Якщо дані сортовані за часом, перевіряємо стрибки
                if isinstance(data.index, pd.DatetimeIndex) and data.index.is_monotonic_increasing:
                    # Криптовалюти можуть мати дуже різкі стрибки
                    pct_change = data[col].pct_change()

                    # Заповнюємо NaN значення (перший елемент) нулем
                    pct_change = pct_change.fillna(0)

                    # Використовуємо різні пороги для крипто (більші ніж для звичайних активів)
                    # Враховуємо як додатні, так і від'ємні стрибки
                    pos_jumps = pct_change > 0.1  # 10% стрибок вгору
                    neg_jumps = pct_change < -0.1  # 10% стрибок вниз
                    price_jumps = pos_jumps | neg_jumps

                    if price_jumps.any():
                        outliers_df[f'{col}_price_jump'] = price_jumps
                        all_outlier_indices.update(data.index[price_jumps])
                        self.logger.info(f"Знайдено {price_jumps.sum()} різких змін ціни у колонці {col}")

                # Перевірка на нульові або від'ємні ціни (неможливо для криптовалют)
                invalid_prices = data[col] <= 0
                if invalid_prices.any():
                    outliers_df[f'{col}_invalid_price'] = invalid_prices
                    all_outlier_indices.update(data.index[invalid_prices])
                    self.logger.info(f"Знайдено {invalid_prices.sum()} нульових або від'ємних значень у колонці {col}")

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
                outliers_df['invalid_ohlc'] = invalid_ohlc
                all_outlier_indices.update(data.index[invalid_ohlc])
                self.logger.info(f"Знайдено {invalid_ohlc.sum()} записів з неузгодженими OHLC даними")

        # 3. Аномалії об'єму торгів (дуже важливо для криптовалют)
        if volume_col:
            if data[volume_col].dtype == object:
                data[volume_col] = pd.to_numeric(data[volume_col], errors='coerce')

            # Перевірка на нульовий об'єм (підозріло для активно торгованих криптовалют)
            zero_volume = data[volume_col] == 0
            if zero_volume.any():
                outliers_df['zero_volume'] = zero_volume
                # Не додаємо до all_outlier_indices, оскільки нульовий об'єм може бути нормальним для деяких періодів
                self.logger.info(f"Знайдено {zero_volume.sum()} записів з нульовим об'ємом")

            # Перевірка на аномально високий об'єм
            # Використовуємо min_periods=1 для обробки коротких даних
            rolling_vol = data[volume_col].rolling(window=24, min_periods=1).median()  # Медіанний об'єм за 24 періоди

            # Уникаємо ділення на нуль
            with np.errstate(divide='ignore', invalid='ignore'):
                vol_ratio = data[volume_col] / rolling_vol
                vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

            high_volume = vol_ratio > 10  # Об'єм у 10+ разів вище медіанного

            if high_volume.any():
                outliers_df['high_volume'] = high_volume
                all_outlier_indices.update(data.index[high_volume])
                self.logger.info(f"Знайдено {high_volume.sum()} записів з аномально високим об'ємом")

            # Від'ємний об'єм - неможливо для криптовалют
            negative_volume = data[volume_col] < 0
            if negative_volume.any():
                outliers_df['negative_volume'] = negative_volume
                all_outlier_indices.update(data.index[negative_volume])
                self.logger.info(f"Знайдено {negative_volume.sum()} записів з від'ємним об'ємом")

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
                self.detect_zscore_outliers(processed_data, numeric_cols, threshold, anomalies_dict,
                                            all_outlier_indices)
            elif method == 'iqr':
                self.detect_iqr_outliers(processed_data, numeric_cols, threshold, anomalies_dict, all_outlier_indices)
            elif method == 'isolation_forest':
                # Для isolation_forest використовуємо contamination замість threshold
                self.detect_isolation_forest_outliers(processed_data, numeric_cols, contamination, anomalies_dict,
                                                      all_outlier_indices)
            elif method == 'crypto_specific':
                # Метод, специфічний для криптовалют
                self.detect_crypto_specific_outliers(processed_data, numeric_cols, threshold, anomalies_dict,
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