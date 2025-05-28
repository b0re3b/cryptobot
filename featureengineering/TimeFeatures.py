from typing import Optional, List
import numpy as np
import pandas as pd

from utils.logger import CryptoLogger


class TimeFeatures:
    def __init__(self):
        self.logger = CryptoLogger('TimeFeatures')

    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Перевіряє та перетворює індекс в DatetimeIndex якщо це можливо.
        """
        if isinstance(data.index, pd.DatetimeIndex):
            return data

        # Спробуємо перетворити індекс в datetime
        try:
            # Якщо індекс числовий, спробуємо трактувати як timestamp
            if pd.api.types.is_numeric_dtype(data.index):
                # Перевіряємо, чи це Unix timestamp
                if data.index.min() > 1e9:  # Приблизно після 2001 року
                    new_index = pd.to_datetime(data.index, unit='s')
                    self.logger.info("Індекс успішно перетворено з Unix timestamp в DatetimeIndex")
                else:
                    # Можливо це timestamp в мілісекундах
                    new_index = pd.to_datetime(data.index, unit='ms')
                    self.logger.info("Індекс успішно перетворено з timestamp (ms) в DatetimeIndex")
            else:
                # Спробуємо стандартне перетворення
                new_index = pd.to_datetime(data.index)
                self.logger.info("Індекс успішно перетворено в DatetimeIndex")

            # Створюємо копію з новим індексом
            result_df = data.copy()
            result_df.index = new_index
            return result_df

        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
            self.logger.warning(f"Не вдалося перетворити індекс в DatetimeIndex: {str(e)}")
            return data

    def _validate_input_data(self, data: pd.DataFrame, method_name: str) -> bool:
        """
        Валідація вхідних даних.
        """
        try:
            # Перевірка на порожній DataFrame
            if data.empty:
                self.logger.error(f"{method_name}: Отримано порожній DataFrame")
                return False

            # Перевірка на наявність NaN в індексі
            if data.index.isna().any():
                self.logger.error(f"{method_name}: Індекс містить NaN значення")
                return False

            # Перевірка на дублікати в індексі
            if data.index.duplicated().any():
                self.logger.warning(f"{method_name}: Індекс містить дублікати")

            return True

        except Exception as e:
            self.logger.error(f"{method_name}: Помилка валідації даних: {str(e)}")
            return False

    def _safe_column_selection(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> List[str]:
        """
        Безпечний вибір числових стовпців з додатковими перевірками.
        """
        try:
            if columns is None:
                # Вибираємо числові стовпці, виключаючи datetime стовпці
                numeric_columns = []
                for col in data.columns:
                    try:
                        # Перевіряємо, чи стовпець числовий і не містить тільки NaN
                        if pd.api.types.is_numeric_dtype(data[col]) and not data[col].isna().all():
                            numeric_columns.append(col)
                    except Exception as e:
                        self.logger.debug(f"Пропускаємо стовпець {col}: {str(e)}")
                        continue

                if not numeric_columns:
                    self.logger.warning("Не знайдено валідних числових стовпців")
                    return []

                self.logger.info(f"Автоматично вибрано {len(numeric_columns)} числових стовпців: {numeric_columns}")
                return numeric_columns
            else:
                # Фільтруємо стовпці, які є в даних та є числовими
                valid_columns = []
                for col in columns:
                    if col not in data.columns:
                        self.logger.warning(f"Стовпець '{col}' не знайдено в даних")
                        continue

                    try:
                        if not pd.api.types.is_numeric_dtype(data[col]):
                            self.logger.warning(f"Стовпець '{col}' не є числовим")
                            continue

                        if data[col].isna().all():
                            self.logger.warning(f"Стовпець '{col}' містить тільки NaN значення")
                            continue

                        valid_columns.append(col)
                    except Exception as e:
                        self.logger.warning(f"Помилка при перевірці стовпця '{col}': {str(e)}")
                        continue

                if len(valid_columns) < len(columns):
                    missing_cols = set(columns) - set(valid_columns)
                    self.logger.warning(f"Стовпці {missing_cols} не будуть використані")

                return valid_columns

        except Exception as e:
            self.logger.error(f"Помилка при виборі стовпців: {str(e)}")
            return []

    def create_lagged_features(self, data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               lag_periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        """
    Векторизовано створює лагові (відкладені у часі) ознаки для заданих стовпців у часовому ряді.

    Лагові ознаки особливо корисні для моделей, що враховують часову залежність,
    таких як моделі прогнозування часових рядів (time series forecasting) або класифікації станів ринку.
    Метод дозволяє створювати кілька лагів одночасно та забезпечує логування та обробку помилок.

    Args:
        data (pd.DataFrame): Вхідний DataFrame з часовим рядом. Бажано, щоб індекс був типу pd.DatetimeIndex.
        columns (Optional[List[str]]): Список стовпців, для яких потрібно створити лаги.
            Якщо None — буде автоматично обрано всі числові стовпці.
        lag_periods (List[int]): Список періодів лагу, які слід згенерувати. Наприклад, [1, 3, 5]
            створить ознаки з відставанням на 1, 3 та 5 кроків відповідно.

    Returns:
        pd.DataFrame: Новий DataFrame з доданими лаговими ознаками.
            Назви нових колонок мають формат `{column_name}_lag_{n}`, де `n` — це значення лагу.

    Raises:
        - Всі помилки логуються, метод не піднімає виключення назовні.
        - У разі помилки повертає копію вхідного DataFrame без змін.

    Примітки:
        - Якщо індекс не є `pd.DatetimeIndex`, буде виведено попередження у логах.
        - Метод автоматично викликає `_ensure_datetime_index()` для перетворення індексу, якщо можливо.
        - Якщо не передано `columns`, будуть використані всі числові колонки (`np.number`).
        - Лаги створюються векторизовано для кращої продуктивності.

    Приклад:
        df = pd.DataFrame(ohlcv_data)
        df_with_lags = feature_generator.create_lagged_features(df, columns=["close", "volume"], lag_periods=[1, 7, 14])
    """
        try:
            self.logger.info("Створення лагових ознак...")

            # Валідація вхідних даних
            if not self._validate_input_data(data, "create_lagged_features"):
                return data.copy()

            # Створюємо копію даних та намагаємося перетворити індекс
            result_df = self._ensure_datetime_index(data.copy())

            # Безпечний вибір стовпців
            columns = self._safe_column_selection(result_df, columns)
            if not columns:
                self.logger.warning("Немає валідних стовпців для створення лагових ознак")
                return result_df

            # Перевіряємо, що індекс часовий для правильного зсуву
            if not isinstance(result_df.index, pd.DatetimeIndex):
                self.logger.warning("Індекс даних не є часовим (DatetimeIndex). Лагові ознаки можуть бути неточними.")

            # Створення лагових ознак з обробкою помилок
            new_features_list = []

            for lag in lag_periods:
                try:
                    # Створюємо лаги для всіх потрібних стовпців одночасно
                    lag_data = result_df[columns].shift(lag)
                    lag_data.columns = [f"{col}_lag_{lag}" for col in columns]
                    new_features_list.append(lag_data)

                    self.logger.debug(f"Створено лаг {lag} для {len(columns)} стовпців")

                except Exception as e:
                    self.logger.error(f"Помилка при створенні лагу {lag}: {str(e)}")
                    continue

            # Об'єднуємо всі нові ознаки
            if new_features_list:
                all_new_features = pd.concat(new_features_list, axis=1)
                result_df = pd.concat([result_df, all_new_features], axis=1)

                num_added_features = len(all_new_features.columns)
                self.logger.info(f"Додано {num_added_features} лагових ознак")
            else:
                self.logger.warning("Не вдалося створити жодної лагової ознаки")

            return result_df

        except Exception as e:
            self.logger.error(f"Критична помилка при створенні лагових ознак: {str(e)}")
            return data.copy()

    def create_rolling_features(self, data: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                window_sizes: List[int] = [5, 10, 20, 50],
                                functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
            Векторизовано створює ознаки на основі ковзного вікна для заданих числових стовпців часових рядів.

            Ці ознаки дозволяють моделі враховувати локальні статистичні характеристики
            за певний період часу, що може суттєво покращити якість прогнозів для
            фінансових чи інших часових даних.

            Args:
                data (pd.DataFrame): Вхідний DataFrame з часовими даними.
                    Індекс бажано повинен бути типу pd.DatetimeIndex.
                columns (Optional[List[str]]): Список стовпців, для яких створюються ознаки.
                    Якщо None — будуть автоматично обрані всі числові колонки.
                window_sizes (List[int]): Список розмірів ковзного вікна. Наприклад, [5, 10, 20].
                functions (List[str]): Список статистичних функцій для обчислення в рамках вікна.
                    Підтримуються: ['mean', 'std', 'min', 'max', 'median', 'sum', 'var',
                    'kurt', 'skew', 'quantile_25', 'quantile_75'].

            Returns:
                pd.DataFrame: Новий DataFrame з доданими ознаками ковзного вікна.
                    Формат назв нових стовпців: `{column}_rolling_{window}_{function}`.

            Raises:
                - Метод не піднімає виключення, всі помилки логуються.
                - У разі фатальної помилки повертається копія вхідного DataFrame без змін.

            Примітки:
                - Якщо індекс не є pd.DatetimeIndex, метод попереджає в логах.
                - Використовується метод `.rolling()` з `min_periods=1`, тому значення обчислюються навіть
                  на початку серії, але можуть бути менш точними.
                - Якщо передані непідтримувані функції — вони автоматично ігноруються.

            Приклад:
                df = pd.DataFrame(ohlcv_data)
                df_with_rolling = feature_generator.create_rolling_features(
                    df, columns=["close", "volume"],
                    window_sizes=[5, 20],
                    functions=["mean", "std", "quantile_25"]
                )
            """
        try:
            self.logger.info("Створення ознак на основі ковзного вікна...")

            # Валідація вхідних даних
            if not self._validate_input_data(data, "create_rolling_features"):
                return data.copy()

            # Створюємо копію даних та намагаємося перетворити індекс
            result_df = self._ensure_datetime_index(data.copy())

            # Безпечний вибір стовпців
            columns = self._safe_column_selection(result_df, columns)
            if not columns:
                self.logger.warning("Немає валідних стовпців для створення ознак ковзного вікна")
                return result_df

            # Перевіряємо, що індекс часовий для правильного розрахунку
            if not isinstance(result_df.index, pd.DatetimeIndex):
                self.logger.warning(
                    "Індекс даних не є часовим (DatetimeIndex). Ознаки ковзного вікна можуть бути неточними.")

            # Словник функцій pandas для ковзного вікна
            func_map = {
                'mean': 'mean',
                'std': 'std',
                'min': 'min',
                'max': 'max',
                'median': 'median',
                'sum': 'sum',
                'var': 'var',
                'kurt': 'kurt',
                'skew': 'skew',
                'quantile_25': lambda x: x.quantile(0.25),
                'quantile_75': lambda x: x.quantile(0.75)
            }

            # Перевіряємо, чи всі функції підтримуються
            valid_functions = [f for f in functions if f in func_map]
            if len(valid_functions) < len(functions):
                unsupported_funcs = set(functions) - set(valid_functions)
                self.logger.warning(f"Функції {unsupported_funcs} не підтримуються і будуть пропущені")
                functions = valid_functions

            if not functions:
                self.logger.warning("Немає валідних функцій для ковзного вікна")
                return result_df

            # Підготовка даних для векторизованої обробки
            new_features_list = []

            # Використовуємо векторизований підхід для кожного розміру вікна
            for window in window_sizes:
                try:
                    # Перевіряємо розмір вікна
                    if window <= 0 or window > len(result_df):
                        self.logger.warning(f"Пропускаємо некоректний розмір вікна: {window}")
                        continue

                    # Отримуємо DataFrame з усіма потрібними числовими стовпцями
                    numeric_data = result_df[columns]

                    # Створюємо об'єкт ковзного вікна для всіх стовпців одночасно
                    rolling_data = numeric_data.rolling(window=window, min_periods=1)

                    for func_name in functions:
                        try:
                            # Отримуємо функцію з мапінгу
                            func = func_map[func_name]

                            # Застосовуємо функцію до всіх стовпців одночасно
                            if callable(func):
                                rolling_result = rolling_data.apply(func, raw=False)
                            else:
                                rolling_result = getattr(rolling_data, func)()

                            # Перейменовуємо стовпці
                            rolling_result.columns = [f"{col}_rolling_{window}_{func_name}" for col in columns]
                            new_features_list.append(rolling_result)

                        except Exception as e:
                            self.logger.error(f"Помилка при обчисленні {func_name} для вікна {window}: {str(e)}")
                            continue

                except Exception as e:
                    self.logger.error(f"Помилка при обробці вікна {window}: {str(e)}")
                    continue

            # Об'єднуємо всі нові ознаки
            if new_features_list:
                all_new_features = pd.concat(new_features_list, axis=1)
                result_df = pd.concat([result_df, all_new_features], axis=1)

                self.logger.info(f"Додано {len(all_new_features.columns)} ознак ковзного вікна")
            else:
                self.logger.warning("Не вдалося створити жодної ознаки ковзного вікна")

            return result_df

        except Exception as e:
            self.logger.error(f"Критична помилка при створенні ознак ковзного вікна: {str(e)}")
            return data.copy()

    def create_ewm_features(self, data: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            spans: List[int] = [5, 10, 20, 50],
                            functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
            Векторизовано створює ознаки на основі експоненційно зваженого вікна (EWM)
            для заданих числових стовпців часових рядів.

            Експоненційне згладжування надає більшу вагу останнім спостереженням,
            що дозволяє краще реагувати на останні зміни в ринку або часі.

            Args:
                data (pd.DataFrame): Вхідний DataFrame з часовими рядами.
                    Бажано, щоб індекс був типу pd.DatetimeIndex.
                columns (Optional[List[str]]): Список стовпців для створення ознак.
                    Якщо None — будуть обрані всі числові стовпці.
                spans (List[int]): Список значень параметра `span` для EWM.
                    Наприклад, [5, 10, 20].
                functions (List[str]): Список статистичних функцій для обчислення в рамках EWM.
                    Підтримуються: ['mean', 'std', 'var'].

            Returns:
                pd.DataFrame: DataFrame з доданими EWM-ознаками.
                    Формат назв нових стовпців: `{column}_ewm_{span}_{function}`.

            Raises:
                - Метод не піднімає виключень, усі помилки логуються.
                - У випадку критичної помилки повертає копію вхідного DataFrame без змін.

            Примітки:
                - Якщо передані непідтримувані функції, вони будуть автоматично виключені.
                - Всі обчислення проводяться векторизовано, що забезпечує високу продуктивність.
                - Значення `span <= 0` ігноруються.

            Приклад:
                df = pd.DataFrame(ohlcv_data)
                df_with_ewm = feature_generator.create_ewm_features(
                    df, columns=["close", "volume"],
                    spans=[10, 20],
                    functions=["mean", "std"]
                )
            """
        try:
            self.logger.info("Створення ознак на основі експоненційно зваженого вікна...")

            # Валідація вхідних даних
            if not self._validate_input_data(data, "create_ewm_features"):
                return data.copy()

            # Створюємо копію даних та намагаємося перетворити індекс
            result_df = self._ensure_datetime_index(data.copy())

            # Безпечний вибір стовпців
            columns = self._safe_column_selection(result_df, columns)
            if not columns:
                self.logger.warning("Немає валідних стовпців для створення EWM ознак")
                return result_df

            # Словник підтримуваних функцій для EWM
            func_map = {
                'mean': 'mean',
                'std': 'std',
                'var': 'var',
            }

            # Перевіряємо, чи всі функції підтримуються
            valid_functions = [f for f in functions if f in func_map]
            if len(valid_functions) < len(functions):
                unsupported_funcs = set(functions) - set(valid_functions)
                self.logger.warning(f"Функції {unsupported_funcs} не підтримуються для EWM і будуть пропущені")
                functions = valid_functions

            if not functions:
                self.logger.warning("Немає валідних функцій для EWM")
                return result_df

            # Підготовка даних для векторизованої обробки
            new_features_list = []

            # Векторизована обробка для кожного span
            for span in spans:
                try:
                    # Перевіряємо span
                    if span <= 0:
                        self.logger.warning(f"Пропускаємо некоректний span: {span}")
                        continue

                    # Створюємо об'єкт експоненційно зваженого вікна для всіх стовпців одночасно
                    ewm_data = result_df[columns].ewm(span=span, min_periods=1)

                    for func_name in functions:
                        try:
                            # Отримуємо функцію з мапінгу і застосовуємо до всіх стовпців
                            func = func_map[func_name]
                            ewm_result = getattr(ewm_data, func)()

                            # Перейменовуємо стовпці
                            ewm_result.columns = [f"{col}_ewm_{span}_{func_name}" for col in columns]
                            new_features_list.append(ewm_result)

                        except Exception as e:
                            self.logger.error(f"Помилка при обчисленні EWM {func_name} для span {span}: {str(e)}")
                            continue

                except Exception as e:
                    self.logger.error(f"Помилка при обробці EWM span {span}: {str(e)}")
                    continue

            # Об'єднуємо всі нові ознаки
            if new_features_list:
                all_new_features = pd.concat(new_features_list, axis=1)
                result_df = pd.concat([result_df, all_new_features], axis=1)

                self.logger.info(f"Додано {len(all_new_features.columns)} ознак експоненційно зваженого вікна")
            else:
                self.logger.warning("Не вдалося створити жодної EWM ознаки")

            return result_df

        except Exception as e:
            self.logger.error(f"Критична помилка при створенні EWM ознак: {str(e)}")
            return data.copy()

    def create_datetime_features(self, data: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
        """
           Векторизовано створює ознаки на основі дати та часу з індексу DatetimeIndex.

           Метод генерує як базові календарні ознаки, так і бінарні, часові сегменти доби, циклічні перетворення,
           та сегментацію торгових годин (наприклад, Asia/Europe/America), що корисно для часових рядів у фінансовому контексті.

           Args:
               data (pd.DataFrame): Вхідний DataFrame із часовим індексом (DatetimeIndex).
               cyclical (bool): Якщо True — створюються циклічні ознаки (sin/cos перетворення для годин, днів, місяців тощо).

           Returns:
               pd.DataFrame: DataFrame з доданими часовими ознаками.

           Raises:
               - Метод не піднімає виключень — усі помилки логуються.
               - У випадку критичної помилки повертається копія початкових даних без змін.

           Додає наступні ознаки:
               - **Базові календарні**:
                   - `hour`, `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `day_of_year`
               - **Бінарні**:
                   - `is_weekend`, `is_month_end`, `is_quarter_end`, `is_year_end`
               - **Сегменти доби**:
                   - `time_of_day` (night, morning, day, evening)
                   - `utc_segment` (asia_late, asia_main, europe_main, america_main, asia_early)
               - **Циклічні (за потреби)**:
                   - `hour_sin`, `hour_cos`
                   - `day_of_week_sin`, `day_of_week_cos`
                   - `day_of_month_sin`, `day_of_month_cos`
                   - `month_sin`, `month_cos`
                   - `day_of_year_sin`, `day_of_year_cos`
               - **Інші**:
                   - `timestamp` — Unix timestamp (в секундах)

           Приклад:
               df = pd.DataFrame(data, index=pd.to_datetime(data["datetime"]))
               df_with_time = feature_generator.create_datetime_features(df, cyclical=True)
           """
        try:
            self.logger.info("Створення ознак на основі дати і часу...")

            # Валідація вхідних даних
            if not self._validate_input_data(data, "create_datetime_features"):
                return data.copy()

            # Створюємо копію даних та намагаємося перетворити індекс
            result_df = self._ensure_datetime_index(data.copy())

            # Перевірити, що індекс є DatetimeIndex
            if not isinstance(result_df.index, pd.DatetimeIndex):
                self.logger.warning("Індекс даних не є DatetimeIndex. Часові ознаки не будуть створені.")
                return result_df

            # DataFrame для зберігання всіх нових ознак
            datetime_features = pd.DataFrame(index=result_df.index)

            try:
                # Базові ознаки (векторизовані)
                datetime_features['hour'] = result_df.index.hour
                datetime_features['day_of_week'] = result_df.index.dayofweek
                datetime_features['day_of_month'] = result_df.index.day
                datetime_features['month'] = result_df.index.month
                datetime_features['quarter'] = result_df.index.quarter
                datetime_features['year'] = result_df.index.year
                datetime_features['day_of_year'] = result_df.index.dayofyear

                # Час доби (ранок, день, вечір, ніч) - векторизована версія
                try:
                    bins = [-1, 5, 11, 17, 23]
                    labels = ['night', 'morning', 'day', 'evening']
                    datetime_features['time_of_day'] = pd.cut(
                        result_df.index.hour,
                        bins=bins,
                        labels=labels
                    ).astype(str)
                except Exception as e:
                    self.logger.warning(f"Помилка при створенні time_of_day: {str(e)}")

                # Бінарні ознаки (векторизовані)
                datetime_features['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
                datetime_features['is_month_end'] = result_df.index.is_month_end.astype(int)
                datetime_features['is_quarter_end'] = result_df.index.is_quarter_end.astype(int)
                datetime_features['is_year_end'] = result_df.index.is_year_end.astype(int)

                # Якщо потрібні циклічні ознаки - створюємо векторизовано
                if cyclical:
                    try:
                        # Циклічні ознаки для години (0-23)
                        datetime_features['hour_sin'] = np.sin(2 * np.pi * datetime_features['hour'] / 24)
                        datetime_features['hour_cos'] = np.cos(2 * np.pi * datetime_features['hour'] / 24)

                        # Циклічні ознаки для дня тижня (0-6)
                        datetime_features['day_of_week_sin'] = np.sin(2 * np.pi * datetime_features['day_of_week'] / 7)
                        datetime_features['day_of_week_cos'] = np.cos(2 * np.pi * datetime_features['day_of_week'] / 7)

                        # Циклічні ознаки для дня місяця (1-31)
                        datetime_features['day_of_month_sin'] = np.sin(
                            2 * np.pi * datetime_features['day_of_month'] / 31)
                        datetime_features['day_of_month_cos'] = np.cos(
                            2 * np.pi * datetime_features['day_of_month'] / 31)

                        # Циклічні ознаки для місяця (1-12)
                        datetime_features['month_sin'] = np.sin(2 * np.pi * datetime_features['month'] / 12)
                        datetime_features['month_cos'] = np.cos(2 * np.pi * datetime_features['month'] / 12)

                        # Циклічні ознаки для дня року (1-366)
                        datetime_features['day_of_year_sin'] = np.sin(
                            2 * np.pi * datetime_features['day_of_year'] / 366)
                        datetime_features['day_of_year_cos'] = np.cos(
                            2 * np.pi * datetime_features['day_of_year'] / 366)
                    except Exception as e:
                        self.logger.warning(f"Помилка при створенні циклічних ознак: {str(e)}")

                # Час доби за UTC (векторизована версія)
                try:
                    bins_utc = [-1, 2, 8, 14, 20, 23]
                    labels_utc = ['asia_late', 'asia_main', 'europe_main', 'america_main', 'asia_early']
                    datetime_features['utc_segment'] = pd.cut(
                        datetime_features['hour'],
                        bins=bins_utc,
                        labels=labels_utc
                    ).astype(str)
                except Exception as e:
                    self.logger.warning(f"Помилка при створенні utc_segment: {str(e)}")

                # Відмітка часу в Unix форматі (векторизована)
                try:
                    datetime_features['timestamp'] = result_df.index.astype(np.int64) // 10 ** 9
                except Exception as e:
                    self.logger.warning(f"Помилка при створенні timestamp: {str(e)}")

                # Додаємо всі ознаки одночасно
                result_df = pd.concat([result_df, datetime_features], axis=1)

                self.logger.info(f"Створено {len(datetime_features.columns)} часових ознак.")

            except Exception as e:
                self.logger.error(f"Помилка при створенні часових ознак: {str(e)}")

            return result_df

        except Exception as e:
            self.logger.error(f"Критична помилка при створенні часових ознак: {str(e)}")
            return data.copy()