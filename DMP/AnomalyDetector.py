from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import decimal
from utils.logger import CryptoLogger
try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnomalyDetector:
    def __init__(self, preserve_original_columns: bool = True):
        """
        Ініціалізація детектора аномалій.

        :param preserve_original_columns: Зберігати всі оригінальні колонки під час обробки
        """
        self.logger = CryptoLogger('INFO')
        self.preserve_original_columns = preserve_original_columns

    def ensure_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """
           Перетворює значення типу decimal.Decimal у float у всіх колонках датафрейму, де це можливо.

           Цей метод проходиться по всіх колонках датафрейму, що мають тип object,
           перевіряє наявність значень типу decimal.Decimal і перетворює їх у float.

           Параметри:
               df (pd.DataFrame): Вхідний датафрейм з можливими decimal.Decimal значеннями.

           Повертає:
               pd.DataFrame: Копія датафрейму, де значення decimal.Decimal замінено на float.

           Логування:
               Інформує про колонки, в яких виконано перетворення типів.
           """
        result = df.copy(deep=True)
        for col in result.columns:
            if result[col].dtype == object:
                has_decimal = any(isinstance(x, decimal.Decimal) for x in result[col].dropna())
                if has_decimal:
                    self.logger.info(f"Converting decimal.Decimal values to float in column {col}")
                    result[col] = result[col].apply(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)
        return result

    def _preprocess_data(self, data: pd.DataFrame, numeric_cols: List[str],
                         fill_method: str = 'median') -> pd.DataFrame:
        """
            Здійснює повну попередню обробку даних криптовалютного датафрейму.

            Операції включають:
                - перетворення текстових значень у числові;
                - заповнення пропущених значень заданим методом;
                - застосування логарифмічного перетворення до сильно скошених числових колонок;
                - видалення дублікатів індексу;
                - сортування за індексом, якщо він є часовим.

            Параметри:
                data (pd.DataFrame): Вхідний датафрейм із сирими даними.
                numeric_cols (List[str]): Список колонок, які вважаються числовими.
                fill_method (str): Метод заповнення пропущених значень: 'mean', 'median', 'ffill', 'bfill' або '0'.

            Повертає:
                pd.DataFrame: Оброблений датафрейм, придатний для подальшого аналізу чи моделювання.

            Логування:
                - Інформує про наявність колонок, виконані перетворення, видалення дублікатів.
                - Зазначає застосування логарифмічних перетворень і їх обґрунтування.
            """
        self.logger.info(f"Наявні колонки в _preprocess_data: {list(data.columns)}")

        # Створюємо глибоку копію з усіма оригінальними колонками
        processed_data = data.copy(deep=True)

        # Перетворення на float з збереженням оригінальних колонок
        for col in processed_data.columns:
            if processed_data[col].dtype == object:
                try:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                except:
                    pass

        if processed_data.empty or not numeric_cols:
            self.logger.warning("Порожні вхідні дані або відсутні числові колонки")
            return processed_data

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
                    fill_value = 0
                processed_data[col] = processed_data[col].fillna(fill_value)

        # 2. Логарифмічне перетворення для криптовалютних даних
        log_columns = []
        for col in numeric_cols:
            # Пропускаємо колонки з одним унікальним значенням
            unique_values = processed_data[col].nunique()
            if unique_values <= 1:
                continue

            # Перевірка на викиди та позитивні значення
            if (processed_data[col] >= 0).all():
                skewness = processed_data[col].skew()
                if abs(skewness) > 1.5:
                    log_col = f'{col}_log'
                    # Додаємо логарифмічну колонку
                    processed_data[log_col] = np.log1p(processed_data[col])
                    log_columns.append(log_col)
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
        """
            Виявлення аномалій у даних з використанням різних методів, оптимізованих для криптовалют.

            Підтримувані методи:
                - 'crypto_specific' (за замовчуванням) — спеціалізований метод для криптовалютних даних;
                - 'zscore' — метод на основі Z-оцінок;
                - 'iqr' — метод на основі міжквартильного діапазону;
                - 'isolation_forest' — метод ізоляційного лісу (потребує scikit-learn);
                - 'ensemble' — ансамбль із кількох методів для підвищення точності.

            Параметри:
                data (pd.DataFrame): Вхідний датафрейм для аналізу.
                method (str): Метод виявлення аномалій.
                threshold (float): Порогове значення для відсіву аномалій (для криптовалют зазвичай більший поріг).
                preprocess (bool): Чи застосовувати попередню обробку даних перед виявленням.
                fill_method (str): Метод заповнення пропущених значень під час передобробки ('mean', 'median', 'ffill', 'bfill' або '0').
                contamination (float): Частка очікуваних аномалій для методу isolation_forest.

            Повертає:
                Tuple[pd.DataFrame, List]:
                    - DataFrame з додатковими колонками, що позначають аномалії для кожного методу та загальним індикатором 'is_anomaly'.
                    - Список індексів рядків, виявлених як аномальні.

            Особливості:
                - Включає логування ключових кроків і помилок.
                - Перетворює типи колонок у числові там, де це можливо.
                - Підтримує ансамблеве поєднання методів для кращої точності.
                - Автоматично обробляє відсутні дані та дублікати індексу.
                - Підвищує поріг для криптовалют через їх високу волатильність.

            Помітки:
                - Метод 'isolation_forest' потребує встановленого пакету scikit-learn.
                - Параметри `threshold` і `contamination` автоматично коригуються при некоректних значеннях.
            """
        self.logger.info(f"Наявні колонки в detect_outliers: {list(data.columns)}")

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для виявлення аномалій")
            return pd.DataFrame(), []

        # Валідація параметрів
        if method not in ['zscore', 'iqr', 'isolation_forest', 'crypto_specific', 'ensemble']:
            self.logger.warning(f"Непідтримуваний метод: {method}. Використовуємо 'crypto_specific'")
            method = 'crypto_specific'

        # Збільшуємо стандартний поріг для криптовалют - вони мають більше природних коливань
        if threshold <= 0:
            self.logger.warning(f"Отримано недопустиме порогове значення: {threshold}. Встановлено значення 5")
            threshold = 5.0

        # Зменшуємо contamination для Isolation Forest для криптовалют - менше помилкових спрацьовувань
        if contamination <= 0 or contamination > 0.5:
            self.logger.warning(f"Неправильне значення contamination: {contamination}. Використовуємо 0.05")
            contamination = 0.05

        self.logger.info(f"Початок виявлення аномалій методом {method} з порогом {threshold}")

        # Власний метод перетворення на float, щоб уникнути помилок
        def ensure_float(df):
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            return df

        data_copy = ensure_float(data.copy(deep=True))

        # Вибір числових колонок
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns.tolist()

        # Перетворення об'єктів на числа, коли можливо
        for col in data_copy.columns:
            if col not in numeric_cols and data_copy[col].dtype == object:
                try:
                    data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                    if not data_copy[col].isna().all():
                        numeric_cols.append(col)
                except:
                    pass

        if len(numeric_cols) == 0:
            self.logger.warning("У DataFrame немає числових колонок для аналізу аномалій")
            return pd.DataFrame(), []

        # Попередня обробка даних
        processed_data = data_copy
        if preprocess:
            try:
                processed_data = self._preprocess_data(data_copy, numeric_cols, fill_method)
                self.logger.info(f"Дані передоброблені з методом заповнення '{fill_method}'")
            except Exception as e:
                self.logger.error(f"Помилка під час передобробки даних: {str(e)}")
                processed_data = data_copy

        # Результати зберігаємо в словнику, з префіксами для індикаторів аномалій
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

                # Перевірка наявності scikit-learn та достатньої кількості даних
                if SKLEARN_AVAILABLE and len(processed_data) >= 50:
                    self._detect_isolation_forest_anomalies(processed_data, numeric_cols, contamination, anomalies,
                                                            all_outlier_indices)

            elif method == 'zscore':
                self._detect_robust_zscore_anomalies(processed_data, numeric_cols, threshold, anomalies,
                                                     all_outlier_indices)

            elif method == 'iqr':
                self._detect_robust_iqr_anomalies(processed_data, numeric_cols, threshold, anomalies,
                                                  all_outlier_indices)

            elif method == 'isolation_forest':
                # Перевірка наявності scikit-learn
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

        # Перетворюємо результати на DataFrame - ВИПРАВЛЕНО для збереження оригінальних колонок
        result_df = data.copy(deep=True)

        # Додаємо інформацію про аномалії без перейменування колонок
        for anomaly_type, indices in anomalies.items():
            if indices:
                # Додаємо префікс anomaly_ до всіх колонок аномалій, якщо його ще немає
                anomaly_col = f"anomaly_{anomaly_type}" if not anomaly_type.startswith("anomaly_") else anomaly_type

                result_df[anomaly_col] = False
                result_df.loc[indices, anomaly_col] = True

        # Додаємо загальний індикатор аномалій
        anomaly_columns = [col for col in result_df.columns if col.startswith('anomaly_')]
        if anomaly_columns:
            result_df['is_anomaly'] = result_df[anomaly_columns].any(axis=1)
        else:
            result_df['is_anomaly'] = False

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return result_df, outlier_indices

    def _detect_robust_zscore_anomalies(self, data: pd.DataFrame, numeric_cols: List[str],
                                        threshold: float, anomalies: Dict[str, List],
                                        all_outlier_indices: set) -> None:
        """
           Виявлення аномалій у числових колонках за допомогою робастного Z-Score.

           Робастний Z-Score розраховується з використанням медіани та медіанного абсолютного відхилення (MAD),
           що є більш стійким до викидів порівняно зі стандартним відхиленням.

           Параметри:
               data (pd.DataFrame): Датафрейм з даними для аналізу.
               numeric_cols (List[str]): Список числових колонок для аналізу.
               threshold (float): Поріг Z-Score для позначення аномалій.
               anomalies (Dict[str, List]): Словник для збереження індексів виявлених аномалій по колонках.
               all_outlier_indices (set): Множина для накопичення унікальних індексів всіх аномалій.

           Особливості:
               - Виключає колонки з порожніми або майже сталими значеннями.
               - Використовує масштабування на основі MAD для криптовалют, де дані можуть бути нестабільними.
               - Логування інформації про знайдені аномалії для кожної колонки.
               - Оновлює передані структури даних аномалій без повернення значення.

           Логування:
               - Виводить інформацію про колонки для аналізу.
               - Повідомляє кількість знайдених аномалій у кожній колонці.
           """
        self.logger.info(f"Наявні колонки в _detect_robust_zscore_anomalies: {list(data.columns)}")
        self.logger.info(f"Числові колонки для аналізу: {numeric_cols}")
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
                # Зберігаємо індекси аномалій для кожної колонки окремо
                anomaly_indices = data.index[outliers].tolist()
                anomalies[f"zscore_{col}"] = anomaly_indices
                outlier_count = outliers.sum()
                self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (zscore)")
                all_outlier_indices.update(anomaly_indices)

    def _detect_robust_iqr_anomalies(self, data: pd.DataFrame, numeric_cols: List[str],
                                     threshold: float, anomalies: Dict[str, List],
                                     all_outlier_indices: set) -> None:
        """
            Виявлення аномалій у числових колонках за методом розширеного міжквартильного діапазону (IQR),
            адаптованого для специфіки криптовалютних даних.

            Метод використовує розширені квантилі (10-й та 90-й перцентили) замість стандартних (25-й та 75-й),
            а також спеціальне оброблення випадків, коли перцентили співпадають.

            Параметри:
                data (pd.DataFrame): Датафрейм з даними для аналізу.
                numeric_cols (List[str]): Список числових колонок для аналізу.
                threshold (float): Поріг множника для визначення меж аномалій.
                anomalies (Dict[str, List]): Словник для збереження індексів виявлених аномалій по колонках.
                all_outlier_indices (set): Множина для накопичення унікальних індексів всіх аномалій.

            Особливості:
                - Виключає колонки з повністю пропущеними або майже сталими значеннями.
                - Для криптовалют використовує більш широкі межі, щоб врахувати високу волатильність.
                - Обробляє випадки, коли стандартні межі IQR неінформативні (Q1 ≈ Q3).
                - Логування інформації про знайдені аномалії по кожній колонці.
                - Оновлює передані структури даних аномалій без повернення значення.

            Логування:
                - Виводить інформацію про колонки для аналізу.
                - Повідомляє кількість знайдених аномалій у кожній колонці.
                - Фіксує помилки при обробці колонок.
            """
        self.logger.info(f"Наявні колонки в _detect_robust_iqr_anomalies: {list(data.columns)}")
        self.logger.info(f"Числові колонки для аналізу: {numeric_cols}")

        for col in numeric_cols:
            if col not in data.columns:
                continue

            # Створюємо копію стовпця з відновленням індексації
            valid_data = pd.to_numeric(data[col], errors='coerce')

            if valid_data.isna().all() or valid_data.nunique() <= 1:
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

                # Створюємо серію з позначками аномалій з повним збереженням індексів
                outliers = (valid_data < lower_bound) | (valid_data > upper_bound)

                if outliers.any():
                    # Зберігаємо індекси аномалій для кожної колонки окремо
                    anomaly_indices = data.index[outliers].tolist()
                    anomalies[f"iqr_{col}"] = anomaly_indices
                    outlier_count = outliers.sum()
                    self.logger.info(f"Знайдено {outlier_count} аномалій у колонці {col} (IQR)")
                    all_outlier_indices.update(anomaly_indices)

            except Exception as e:
                self.logger.error(f"Помилка при застосуванні IQR методу до колонки {col}: {str(e)}")

    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame, numeric_cols: List[str],
                                           contamination: float, anomalies: Dict[str, List],
                                           all_outlier_indices: set) -> None:
        """
    Виявлення аномалій у числових колонках за допомогою методу Isolation Forest,
    з оптимізацією параметрів під особливості криптовалютних даних.

    Метод:
        - Перевіряє наявність необхідної кількості даних (рекомендовано >= 20 записів).
        - Очищує та підготовлює числові колонки, усуваючи константні.
        - Заповнює пропущені значення медіаною для стабільності.
        - Налаштовує Isolation Forest з більшою кількістю дерев та бутстрепом для кращої робусності.
        - Логікує та зберігає індекси виявлених аномалій.

    Параметри:
        data (pd.DataFrame): Вхідний датафрейм з даними.
        numeric_cols (List[str]): Список числових колонок для аналізу.
        contamination (float): Очікувана частка аномалій у вибірці.
        anomalies (Dict[str, List]): Словник для збереження індексів аномалій.
        all_outlier_indices (set): Множина для накопичення унікальних індексів аномалій.

    Логування:
        - Повідомляє про наявність колонок і кількість знайдених аномалій.
        - Фіксує помилки при виконанні.
    """
        self.logger.info(f"Наявні колонки в _detect_isolation_forest_anomalies: {list(data.columns)}")
        self.logger.info(f"Числові колонки для аналізу: {numeric_cols}")
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
                # Зберігаємо загальні аномалії, знайдені Isolation Forest
                anomalies["isolation_forest"] = anomaly_indices
                anomaly_count = len(anomaly_indices)
                self.logger.info(f"Знайдено {anomaly_count} аномалій методом Isolation Forest")
                all_outlier_indices.update(anomaly_indices)

        except Exception as e:
            self.logger.error(f"Помилка при використанні Isolation Forest: {str(e)}")

    def _detect_crypto_specific_anomalies(self, data: pd.DataFrame, threshold: float,
                                          anomalies: Dict[str, List], all_outlier_indices: set) -> None:
        """
            Спеціалізований метод для виявлення аномалій, характерних для криптовалютних даних.

            Виявляє:
                - Різкі стрибки цін (>15% за період) у колонках open, high, low, close.
                - Нульові або від’ємні значення цін.
                - Неузгодженості в OHLC (open, high, low, close) даних.
                - Аномальні значення об’єму торгів (нульові, від’ємні, аномально високі).
                - Flash crash та flash pump події на основі амплітуди коливань.
                - Потенційні маніпуляції з flash crash/pump (швидке відновлення в межах свічки).
                - Зріджені часові дані (великі пропуски між свічками).

            Особливості:
                - Для обчислень використовує адаптовані пороги, враховуючи волатильність криптовалют.
                - Логування інформації про знайдені аномалії з детальним описом.
                - Деякі типи аномалій (наприклад, нульовий об’єм) логуються, але не додаються до загального списку аномалій.

            Параметри:
                data (pd.DataFrame): Вхідний датафрейм з криптовалютними даними.
                threshold (float): Поріг, що використовується для деяких видів аналізу.
                anomalies (Dict[str, List]): Словник для збереження індексів аномалій.
                all_outlier_indices (set): Множина для накопичення унікальних індексів аномалій.

            Логування:
                - Детальна інформація по кожному виду аномалій.
                - Повідомлення про потенційні помилки.
            """
        self.logger.info(f"Наявні колонки в _detect_crypto_specific_anomalies: {list(data.columns)}")
        # Перевірка наявності типових колонок для криптовалютних даних
        crypto_price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        volume_col = 'volume' if 'volume' in data.columns else None

        if not crypto_price_cols and not volume_col:
            self.logger.warning("Не знайдено типових колонок для криптовалютних даних")
            return

        # Створюємо копію даних для обробки, щоб не змінювати оригінальний DataFrame
        processed_data = data.copy()

        # 1. Виявлення різких стрибків цін - специфічно для криптовалют використовуємо більші пороги
        if len(crypto_price_cols) > 0:
            for col in crypto_price_cols:
                if processed_data[col].dtype == object:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

                # Якщо дані сортовані за часом, перевіряємо стрибки
                if isinstance(processed_data.index, pd.DatetimeIndex) and processed_data.index.is_monotonic_increasing:
                    pct_change = processed_data[col].pct_change().fillna(0)

                    # Для криптовалют використовуємо більший поріг - 15% (замість 10%)
                    large_pos_jumps = pct_change > 0.15  # 15% стрибок вгору
                    large_neg_jumps = pct_change < -0.15  # 15% стрибок вниз

                    price_jumps = large_pos_jumps | large_neg_jumps

                    if price_jumps.any():
                        jump_indices = processed_data.index[price_jumps].tolist()
                        anomalies[f"price_jump_{col}"] = jump_indices
                        self.logger.info(f"Знайдено {price_jumps.sum()} різких змін ціни у колонці {col}")
                        all_outlier_indices.update(jump_indices)

                # Перевірка на нульові ціни (неможливо для більшості криптовалют)
                invalid_prices = processed_data[col] <= 0
                if invalid_prices.any():
                    invalid_indices = processed_data.index[invalid_prices].tolist()
                    anomalies[f"invalid_price_{col}"] = invalid_indices
                    self.logger.info(f"Знайдено {invalid_prices.sum()} нульових або від'ємних значень у колонці {col}")
                    all_outlier_indices.update(invalid_indices)

        # 2. Перевірка узгодженості OHLC даних
        if all(col in processed_data.columns for col in ['open', 'high', 'low', 'close']):
            # Перетворення на числові дані
            for col in ['open', 'high', 'low', 'close']:
                if processed_data[col].dtype == object:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

            # High має бути >= всіх інших цін
            invalid_high = (processed_data['high'] < processed_data['open']) | (
                    processed_data['high'] < processed_data['low']) | (
                                   processed_data['high'] < processed_data['close'])

            # Low має бути <= всіх інших цін
            invalid_low = (processed_data['low'] > processed_data['open']) | (
                    processed_data['low'] > processed_data['high']) | (
                                  processed_data['low'] > processed_data['close'])

            invalid_ohlc = invalid_high | invalid_low

            if invalid_ohlc.any():
                invalid_indices = processed_data.index[invalid_ohlc].tolist()
                anomalies["invalid_ohlc"] = invalid_indices
                self.logger.info(f"Знайдено {invalid_ohlc.sum()} записів з неузгодженими OHLC даними")
                all_outlier_indices.update(invalid_indices)

        # 3. Аномалії об'єму торгів (дуже важливо для криптовалют)
        if volume_col:
            if processed_data[volume_col].dtype == object:
                processed_data[volume_col] = pd.to_numeric(processed_data[volume_col], errors='coerce')

            # Перевірка на нульовий об'єм - ВИПРАВЛЕНО
            # Тепер нульовий об'єм лише логується, але не додається автоматично до аномалій
            zero_volume = processed_data[volume_col] == 0
            if zero_volume.any():
                zero_vol_indices = processed_data.index[zero_volume].tolist()
                anomalies["zero_volume"] = zero_vol_indices
                # Тепер НЕ додаємо нульові обсяги до загальних аномалій
                self.logger.info(f"Знайдено {zero_volume.sum()} записів з нульовим об'ємом (не обов'язково аномалія)")
                # Закоментовано: all_outlier_indices.update(zero_vol_indices)

            # Перевірка на аномально високий об'єм - криптовалюти можуть мати великі стрибки об'єму
            # Використовуємо тільки ненульові значення для обчислення базової лінії
            non_zero_volume = processed_data[volume_col][processed_data[volume_col] > 0]

            if len(non_zero_volume) > 0:
                # Обчислюємо медіанний об'єм на основі ненульових значень
                rolling_vol = non_zero_volume.rolling(window=min(24, len(non_zero_volume)), min_periods=1).median()

                # Створюємо серію для всіх значень
                vol_ratio = pd.Series(np.nan, index=processed_data.index)

                # Заповнюємо тільки для ненульових індексів
                with np.errstate(divide='ignore', invalid='ignore'):
                    for idx in non_zero_volume.index:
                        if idx in rolling_vol.index and rolling_vol[idx] > 0:
                            vol_ratio[idx] = non_zero_volume[idx] / rolling_vol[idx]

                # Замінюємо NaN та нескінченності
                vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

                # Для криптовалют підвищуємо поріг до 20x
                high_volume = vol_ratio > 20  # Об'єм у 20+ разів вище медіанного

                if high_volume.any():
                    high_vol_indices = processed_data.index[high_volume].tolist()
                    anomalies["high_volume"] = high_vol_indices
                    self.logger.info(f"Знайдено {high_volume.sum()} записів з аномально високим об'ємом")
                    all_outlier_indices.update(high_vol_indices)

            # Від'ємний об'єм - неможливо для криптовалют
            negative_volume = processed_data[volume_col] < 0
            if negative_volume.any():
                neg_vol_indices = processed_data.index[negative_volume].tolist()
                anomalies["negative_volume"] = neg_vol_indices
                self.logger.info(f"Знайдено {negative_volume.sum()} записів з від'ємним об'ємом")
                all_outlier_indices.update(neg_vol_indices)

        # 4. Виявлення Flash Crash та Flash Pump (специфічно для крипторинків)
        if all(col in processed_data.columns for col in ['high', 'low']) and len(processed_data) > 10:
            try:
                # Обчислюємо амплітуду коливань (high-low)/(0.5*(high+low))
                # ВИПРАВЛЕНО: додана перевірка на нульові ціни щоб уникнути ділення на нуль
                denominator = 0.5 * (processed_data['high'] + processed_data['low'])
                # Запобігання діленню на нуль
                valid_denom = denominator > 0

                # Ініціалізуємо масив для амплітуди з NaN
                amplitude = pd.Series(np.nan, index=processed_data.index)

                # Заповнюємо тільки для валідних знаменників
                amplitude[valid_denom] = (processed_data['high'] - processed_data['low'])[valid_denom] / denominator[
                    valid_denom]

                # Знаходимо 95-й перцентиль для амплітуди - це базовий рівень для виявлення flash crash/pump
                perc_95 = amplitude.quantile(0.95, interpolation='linear')

                # Flash crash/pump визначається як амплітуда, що перевищує 95-й перцентиль в 3 рази
                flash_events = amplitude > (perc_95 * 3)

                if flash_events.any():
                    flash_indices = processed_data.index[flash_events].tolist()
                    anomalies["flash_event"] = flash_indices
                    self.logger.info(f"Знайдено {flash_events.sum()} записів з flash crash/pump подіями")
                    all_outlier_indices.update(flash_indices)

                # Додатково перевіряємо на наявність повного відновлення після flash crash/pump в межах свічки
                # Якщо high та close майже рівні після значного падіння, або low та close майже рівні після злету,
                # це може свідчити про маніпуляцію ринком
                if 'close' in processed_data.columns:
                    # Визначення відносної різниці між high і close
                    high_close_diff = (processed_data['high'] - processed_data['close']) / processed_data['high']

                    # Визначення відносної різниці між close і low
                    close_low_diff = (processed_data['close'] - processed_data['low']) / processed_data['close']

                    # Потенційні flash crash з відновленням: великий спад і закриття близько до high
                    flash_crash_recovery = (amplitude > perc_95 * 2) & (high_close_diff < 0.01)

                    # Потенційні flash pump з відкатом: великий стрибок і закриття близько до low
                    flash_pump_reversal = (amplitude > perc_95 * 2) & (close_low_diff < 0.01)

                    # Об'єднуємо обидва типи подій
                    flash_manipulation = flash_crash_recovery | flash_pump_reversal

                    if flash_manipulation.any():
                        flash_manip_indices = processed_data.index[flash_manipulation].tolist()
                        anomalies["flash_manipulation"] = flash_manip_indices
                        self.logger.info(f"Знайдено {flash_manipulation.sum()} записів з підозрою на flash маніпуляції")
                        all_outlier_indices.update(flash_manip_indices)

            except Exception as e:
                self.logger.error(f"Помилка при виявленні flash crash/pump: {str(e)}")

        # 5. Виявлення зріджених даних (рідкісні або відсутні свічки - специфічна проблема для деяких крипто бірж)
        if isinstance(processed_data.index, pd.DatetimeIndex) and len(processed_data) > 10:
            try:
                # Обчислюємо типовий інтервал між свічками
                time_diffs = processed_data.index.to_series().diff().dropna()

                # Якщо є хоча б кілька значень для розрахунку
                if len(time_diffs) > 3:
                    # Знаходимо типовий інтервал (найчастіший)
                    typical_interval = time_diffs.mode()[0]

                    # Знаходимо аномально великі інтервали (в 3+ рази більші за типовий)
                    large_gaps = time_diffs > (typical_interval * 3)

                    if large_gaps.any():
                        # Знаходимо індекси записів ПІСЛЯ великих проміжків
                        gap_indices = time_diffs[large_gaps].index.tolist()
                        anomalies["time_gap"] = gap_indices
                        self.logger.info(
                            f"Знайдено {large_gaps.sum()} записів після аномально великих часових проміжків")
                        # Не додаємо до all_outlier_indices, оскільки це скоріше інформаційна аномалія

            except Exception as e:
                self.logger.error(f"Помилка при виявленні часових проміжків: {str(e)}")



    def validate_datetime_index(self, data: pd.DataFrame, issues: Dict[str, Any]) -> None:
        """Перевірка часового індексу на проблеми."""
        self.logger.info(f"Наявні колонки в validate_datetime_index: {list(data.columns)}")
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
        self.logger.info(f"Наявні колонки в validate_price_data: {list(data.columns)}")
        # Створюємо копію для обробки
        processed_data = data.copy()

        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in processed_data.columns]

        # Переконуємося, що всі цінові колонки числові
        for col in price_cols:
            if processed_data[col].dtype == object:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

        if len(price_cols) == 4:
            # Перевірка на некоректні співвідношення high і low
            invalid_hl = processed_data['high'] < processed_data['low']
            if invalid_hl.any():
                invalid_hl_indices = processed_data.index[invalid_hl].tolist()
                issues["invalid_high_low"] = invalid_hl_indices
                self.logger.warning(f"Знайдено {len(invalid_hl_indices)} записів де high < low")

            # Перевірка на неконсистентність OHLC
            invalid_ohlc = (
                    (processed_data['high'] < processed_data['open']) |
                    (processed_data['high'] < processed_data['close']) |
                    (processed_data['low'] > processed_data['open']) |
                    (processed_data['low'] > processed_data['close'])
            )
            if invalid_ohlc.any():
                invalid_ohlc_indices = processed_data.index[invalid_ohlc].tolist()
                issues["inconsistent_ohlc"] = invalid_ohlc_indices
                self.logger.warning(f"Знайдено {len(invalid_ohlc_indices)} записів з неконсистентними OHLC даними")

            # Для даних Binance - перевірка на нульові ціни (замість від'ємних)
            for col in price_cols:
                zero_prices = processed_data[col] == 0
                if zero_prices.any():
                    zero_price_indices = processed_data.index[zero_prices].tolist()
                    issues[f"zero_{col}"] = zero_price_indices
                    self.logger.warning(f"Знайдено {len(zero_price_indices)} записів з нульовими значеннями у {col}")

            # Перевірка різких стрибків цін
            for col in price_cols:
                try:
                    # Безпечне обчислення відсоткової зміни з обробкою NaN
                    valid_data = processed_data[col].dropna()
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
        self.logger.info(f"Наявні колонки в validate_volume_data: {list(data.columns)}")
        # Створюємо копію для обробки
        processed_data = data.copy()

        if 'volume' in processed_data.columns:
            if processed_data['volume'].dtype == object:
                processed_data['volume'] = pd.to_numeric(processed_data['volume'], errors='coerce')

            # Для даних Binance - перевірка на нульовий об'єм замість від'ємного
            zero_volume = processed_data['volume'] == 0
            if zero_volume.any():
                zero_vol_indices = processed_data.index[zero_volume].tolist()
                issues["zero_volume"] = zero_vol_indices
                self.logger.warning(f"Знайдено {len(zero_vol_indices)} записів з нульовим об'ємом")

            # Перевірка аномального об'єму
            try:
                valid_volume = processed_data['volume'].dropna()
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
        self.logger.info(f"Наявні колонки в validate_data_values: {list(data.columns)}")
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
        self.logger.info(f"Наявні колонки в detect_outliers_essemble: {list(data.columns)}")
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

        # Створюємо копію даних, щоб не змінювати оригінал
        processed_data = data.copy()

        processed_data = self.ensure_float(processed_data)

        # Вибір числових колонок
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning("У DataFrame немає числових колонок для аналізу аномалій")
            return pd.DataFrame(), []

        # Попередня обробка даних
        if preprocess:
            try:
                processed_data = self._preprocess_data(processed_data, numeric_cols, fill_method)
                self.logger.info(f"Дані передоброблені з методом заповнення '{fill_method}'")
            except Exception as e:
                self.logger.error(f"Помилка під час передобробки даних: {str(e)}")
                processed_data = data.copy()  # Повернемося до початкової копії

        # Використовуємо словник для зберігання аномалій замість додавання колонок з суфіксом _outlier
        anomalies_dict = {}
        all_outlier_indices = set()

        try:
            if method == 'zscore':
                self._detect_robust_zscore_anomalies(processed_data, numeric_cols, threshold, anomalies_dict,
                                                     all_outlier_indices)
            elif method == 'iqr':
                self._detect_robust_iqr_anomalies(processed_data, numeric_cols, threshold, anomalies_dict,
                                                  all_outlier_indices)
            elif method == 'isolation_forest':
                # Для isolation_forest використовуємо contamination замість threshold
                self._detect_isolation_forest_anomalies(processed_data, numeric_cols, contamination, anomalies_dict,
                                                        all_outlier_indices)
            elif method == 'crypto_specific':
                # Метод, специфічний для криптовалют
                self._detect_crypto_specific_anomalies(processed_data, threshold, anomalies_dict,
                                                       all_outlier_indices)
            else:
                self.logger.error(f"Непідтримуваний метод виявлення аномалій: {method}")
                return pd.DataFrame(), []

        except Exception as e:
            self.logger.error(f"Помилка під час виявлення аномалій методом {method}: {str(e)}")
            return pd.DataFrame(), []

        # Створюємо результуючий DataFrame з аномаліями - зберігаємо структуру оригінального DataFrame
        outliers_df = pd.DataFrame(index=data.index)

        # Додаємо виявлені аномалії до результуючого DataFrame як окремі колонки
        for col_name, anomalies in anomalies_dict.items():
            # Створюємо колонки типу is_anomaly_[тип_аномалії] замість переписування оригінальних
            outliers_df[f"is_anomaly_{col_name}"] = pd.Series(False, index=data.index)
            outliers_df.loc[anomalies, f"is_anomaly_{col_name}"] = True

        # Додаємо загальну колонку is_outlier
        if not outliers_df.empty:
            outliers_df['is_outlier'] = outliers_df.any(axis=1)

        outlier_indices = list(all_outlier_indices)

        self.logger.info(f"Виявлення аномалій завершено. Знайдено {len(outlier_indices)} аномалій у всіх колонках")
        return outliers_df, outlier_indices