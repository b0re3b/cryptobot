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

    def create_lagged_features(self, data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               lag_periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        """
        Векторизоване створення лагових ознак для часових рядів.
        """
        self.logger.info("Створення лагових ознак...")

        # Створюємо копію даних та намагаємося перетворити індекс
        result_df = self._ensure_datetime_index(data.copy())

        # Перевіряємо, що індекс часовий для правильного зсуву
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є часовим (DatetimeIndex). Лагові ознаки можуть бути неточними.")

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Фільтруємо стовпці, які є в даних
            valid_columns = [col for col in columns if col in result_df.columns]
            if len(valid_columns) < len(columns):
                missing_cols = set(columns) - set(valid_columns)
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
            columns = valid_columns

        # Векторизоване створення лагових ознак
        for lag in lag_periods:
            # Створюємо лаги для всіх потрібних стовпців одночасно
            lag_columns = {f"{col}_lag_{lag}": result_df[col].shift(lag) for col in columns}

            # Оптимізоване додавання стовпців
            result_df = pd.concat([result_df, pd.DataFrame(lag_columns)], axis=1)

            self.logger.debug(f"Створено лаг {lag} для {len(columns)} стовпців")

        # Інформуємо про кількість доданих ознак
        num_added_features = len(columns) * len(lag_periods)
        self.logger.info(f"Додано {num_added_features} лагових ознак")

        return result_df

    def create_rolling_features(self, data: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                window_sizes: List[int] = [5, 10, 20, 50],
                                functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Векторизоване створення ознак на основі ковзного вікна.
        """
        self.logger.info("Створення ознак на основі ковзного вікна...")

        # Створюємо копію даних та намагаємося перетворити індекс
        result_df = self._ensure_datetime_index(data.copy())

        # Перевіряємо, що індекс часовий для правильного розрахунку
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning(
                "Індекс даних не є часовим (DatetimeIndex). Ознаки ковзного вікна можуть бути неточними.")

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Фільтруємо стовпці, які є в даних
            valid_columns = [col for col in columns if col in result_df.columns]
            if len(valid_columns) < len(columns):
                missing_cols = set(columns) - set(valid_columns)
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
            columns = valid_columns

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

        # Підготовка даних для векторизованої обробки
        new_features_dict = {}

        # Використовуємо векторизований підхід для кожного розміру вікна
        for window in window_sizes:
            # Отримуємо DataFrame з усіма потрібними числовими стовпцями
            numeric_data = result_df[columns]

            # Створюємо об'єкт ковзного вікна для всіх стовпців одночасно
            rolling_data = numeric_data.rolling(window=window, min_periods=1)

            for func_name in functions:
                # Отримуємо функцію з мапінгу
                func = func_map[func_name]

                # Застосовуємо функцію до всіх стовпців одночасно
                if callable(func):
                    result = rolling_data.apply(func)
                else:
                    result = getattr(rolling_data, func)()

                # Перейменовуємо стовпці
                result.columns = [f"{col}_rolling_{window}_{func_name}" for col in columns]

                # Додаємо результати до словника нових ознак
                for col in result.columns:
                    new_features_dict[col] = result[col]

        # Додаємо всі нові ознаки до результату
        result_df = pd.concat([result_df, pd.DataFrame(new_features_dict)], axis=1)

        self.logger.info(f"Додано {len(new_features_dict)} ознак ковзного вікна")

        return result_df

    def create_ewm_features(self, data: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            spans: List[int] = [5, 10, 20, 50],
                            functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Векторизоване створення ознак на основі експоненційно зваженого вікна.
        """
        self.logger.info("Створення ознак на основі експоненціально зваженого вікна...")

        # Створюємо копію даних та намагаємося перетворити індекс
        result_df = self._ensure_datetime_index(data.copy())

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Фільтруємо стовпці, які є в даних
            valid_columns = [col for col in columns if col in result_df.columns]
            if len(valid_columns) < len(columns):
                missing_cols = set(columns) - set(valid_columns)
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
            columns = valid_columns

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

        # Підготовка даних для векторизованої обробки
        new_features_dict = {}

        # Векторизована обробка для кожного span
        for span in spans:
            # Створюємо об'єкт експоненціально зваженого вікна для всіх стовпців одночасно
            ewm_data = result_df[columns].ewm(span=span, min_periods=1)

            for func_name in functions:
                # Отримуємо функцію з мапінгу і застосовуємо до всіх стовпців
                func = func_map[func_name]
                result = getattr(ewm_data, func)()

                # Перейменовуємо стовпці
                result.columns = [f"{col}_ewm_{span}_{func_name}" for col in columns]

                # Додаємо результати до словника нових ознак
                for col in result.columns:
                    new_features_dict[col] = result[col]

        # Додаємо всі нові ознаки до результату
        result_df = pd.concat([result_df, pd.DataFrame(new_features_dict)], axis=1)

        self.logger.info(f"Додано {len(new_features_dict)} ознак експоненціально зваженого вікна")

        return result_df

    def create_datetime_features(self, data: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
        """
        Векторизоване створення ознак на основі дати й часу.
        """
        self.logger.info("Створення ознак на основі дати і часу...")

        # Створюємо копію даних та намагаємося перетворити індекс
        result_df = self._ensure_datetime_index(data.copy())

        # Перевірити, що індекс є DatetimeIndex
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є DatetimeIndex. Часові ознаки не будуть створені.")
            return result_df

        # DataFrame для зберігання всіх нових ознак
        datetime_features = pd.DataFrame(index=result_df.index)

        # Базові ознаки (векторизовані)
        datetime_features['hour'] = result_df.index.hour
        datetime_features['day_of_week'] = result_df.index.dayofweek
        datetime_features['day_of_month'] = result_df.index.day
        datetime_features['month'] = result_df.index.month
        datetime_features['quarter'] = result_df.index.quarter
        datetime_features['year'] = result_df.index.year
        datetime_features['day_of_year'] = result_df.index.dayofyear

        # Час доби (ранок, день, вечір, ніч) - векторизована версія
        bins = [-1, 5, 11, 17, 23]
        labels = ['night', 'morning', 'day', 'evening']
        datetime_features['time_of_day'] = pd.cut(
            result_df.index.hour,
            bins=bins,
            labels=labels
        ).astype(str)

        # Бінарні ознаки (векторизовані)
        datetime_features['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
        datetime_features['is_month_end'] = result_df.index.is_month_end.astype(int)
        datetime_features['is_quarter_end'] = result_df.index.is_quarter_end.astype(int)
        datetime_features['is_year_end'] = result_df.index.is_year_end.astype(int)

        # Якщо потрібні циклічні ознаки - створюємо векторизовано
        if cyclical:
            # Циклічні ознаки для години (0-23)
            datetime_features['hour_sin'] = np.sin(2 * np.pi * datetime_features['hour'] / 24)
            datetime_features['hour_cos'] = np.cos(2 * np.pi * datetime_features['hour'] / 24)

            # Циклічні ознаки для дня тижня (0-6)
            datetime_features['day_of_week_sin'] = np.sin(2 * np.pi * datetime_features['day_of_week'] / 7)
            datetime_features['day_of_week_cos'] = np.cos(2 * np.pi * datetime_features['day_of_week'] / 7)

            # Циклічні ознаки для дня місяця (1-31)
            datetime_features['day_of_month_sin'] = np.sin(2 * np.pi * datetime_features['day_of_month'] / 31)
            datetime_features['day_of_month_cos'] = np.cos(2 * np.pi * datetime_features['day_of_month'] / 31)

            # Циклічні ознаки для місяця (1-12)
            datetime_features['month_sin'] = np.sin(2 * np.pi * datetime_features['month'] / 12)
            datetime_features['month_cos'] = np.cos(2 * np.pi * datetime_features['month'] / 12)

            # Циклічні ознаки для дня року (1-366)
            datetime_features['day_of_year_sin'] = np.sin(2 * np.pi * datetime_features['day_of_year'] / 366)
            datetime_features['day_of_year_cos'] = np.cos(2 * np.pi * datetime_features['day_of_year'] / 366)

        # Час доби за UTC (векторизована версія)
        bins_utc = [-1, 2, 8, 14, 20, 23]
        labels_utc = ['asia_late', 'asia_main', 'europe_main', 'america_main', 'asia_early']
        datetime_features['utc_segment'] = pd.cut(
            datetime_features['hour'],
            bins=bins_utc,
            labels=labels_utc
        ).astype(str)

        # Відмітка часу в Unix форматі (векторизована)
        datetime_features['timestamp'] = result_df.index.astype(np.int64) // 10 ** 9

        # Додаємо всі ознаки одночасно
        result_df = pd.concat([result_df, datetime_features], axis=1)

        self.logger.info(f"Створено {len(datetime_features.columns)} часових ознак.")
        return result_df