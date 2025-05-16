from typing import Optional, List

import numpy as np
import pandas as pd

from featureengineering.feature_engineering import FeatureEngineering


class TimeFeatures(FeatureEngineering):
    def create_lagged_features(self, data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               lag_periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        self.logger.info("Створення лагових ознак...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що індекс часовий для правильного зсуву
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є часовим (DatetimeIndex). Лагові ознаки можуть бути неточними.")

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних
            missing_cols = [col for col in columns if col not in result_df.columns]
            if missing_cols:
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
                columns = [col for col in columns if col in result_df.columns]

        # Створюємо лагові ознаки для кожного стовпця і періоду
        for col in columns:
            for lag in lag_periods:
                lag_col_name = f"{col}_lag_{lag}"
                result_df[lag_col_name] = result_df[col].shift(lag)
                self.logger.debug(f"Створено лаг {lag} для стовпця {col}")

        # Інформуємо про кількість доданих ознак
        num_added_features = len(columns) * len(lag_periods)
        self.logger.info(f"Додано {num_added_features} лагових ознак")

        return result_df

    def create_rolling_features(self, data: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                window_sizes: List[int] = [5, 10, 20, 50],
                                functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:

        self.logger.info("Створення ознак на основі ковзного вікна...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що індекс часовий для правильного розрахунку
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning(
                "Індекс даних не є часовим (DatetimeIndex). Ознаки ковзного вікна можуть бути неточними.")

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних
            missing_cols = [col for col in columns if col not in result_df.columns]
            if missing_cols:
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
                columns = [col for col in columns if col in result_df.columns]

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
        unsupported_funcs = [f for f in functions if f not in func_map]
        if unsupported_funcs:
            self.logger.warning(f"Функції {unsupported_funcs} не підтримуються і будуть пропущені")
            functions = [f for f in functions if f in func_map]

        # Лічильник доданих ознак
        added_features_count = 0

        # Для кожного стовпця, розміру вікна і функції створюємо нову ознаку
        for col in columns:
            for window in window_sizes:
                # Створюємо об'єкт ковзного вікна
                rolling_window = result_df[col].rolling(window=window, min_periods=1)

                for func_name in functions:
                    # Отримуємо функцію з мапінгу
                    func = func_map[func_name]

                    # Створюємо нову ознаку
                    feature_name = f"{col}_rolling_{window}_{func_name}"

                    # Застосовуємо функцію до ковзного вікна
                    if callable(func):
                        result_df[feature_name] = rolling_window.apply(func)
                    else:
                        result_df[feature_name] = getattr(rolling_window, func)()

                    added_features_count += 1
                    self.logger.debug(f"Створено ознаку {feature_name}")

        # Обробляємо NaN значення на початку часового ряду
        # Заповнюємо перші значення медіаною стовпця (можна змінити на інший метод)
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                result_df[col] = result_df[col].fillna(result_df[col].median())

        self.logger.info(f"Додано {added_features_count} ознак ковзного вікна")

        return result_df

    def create_ewm_features(self, data: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            spans: List[int] = [5, 10, 20, 50],
                            functions: List[str] = ['mean', 'std']) -> pd.DataFrame:

        self.logger.info("Створення ознак на основі експоненціально зваженого вікна...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Вибираємо числові стовпці, якщо columns не вказано
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Автоматично вибрано {len(columns)} числових стовпців")
        else:
            # Перевіряємо наявність вказаних стовпців у даних
            missing_cols = [col for col in columns if col not in result_df.columns]
            if missing_cols:
                self.logger.warning(f"Стовпці {missing_cols} не знайдено в даних і будуть пропущені")
                columns = [col for col in columns if col in result_df.columns]

        # Словник підтримуваних функцій для EWM
        func_map = {
            'mean': 'mean',
            'std': 'std',
            'var': 'var',
        }

        # Перевіряємо, чи всі функції підтримуються
        unsupported_funcs = [f for f in functions if f not in func_map]
        if unsupported_funcs:
            self.logger.warning(f"Функції {unsupported_funcs} не підтримуються для EWM і будуть пропущені")
            functions = [f for f in functions if f in func_map]

        # Лічильник доданих ознак
        added_features_count = 0

        # Для кожного стовпця, span і функції створюємо нову ознаку
        for col in columns:
            for span in spans:
                # Перевіряємо наявність пропущених значень в стовпці
                if result_df[col].isna().any():
                    self.logger.warning(f"Стовпець {col} містить NaN значення, вони будуть заповнені")
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                # Створюємо об'єкт експоненціально зваженого вікна
                ewm_window = result_df[col].ewm(span=span, min_periods=1)

                for func_name in functions:
                    # Отримуємо функцію з мапінгу
                    func = func_map[func_name]

                    # Створюємо нову ознаку
                    feature_name = f"{col}_ewm_{span}_{func_name}"

                    # Застосовуємо функцію до EWM
                    result_df[feature_name] = getattr(ewm_window, func)()

                    added_features_count += 1
                    self.logger.debug(f"Створено ознаку {feature_name}")

        # Перевіряємо наявність NaN значень у нових ознаках
        for col in result_df.columns:
            if col not in data.columns:  # перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    self.logger.debug(f"Заповнення NaN значень у стовпці {col}")
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                    # Якщо все ще є NaN (можливо на початку), заповнюємо медіаною
                    if result_df[col].isna().any():
                        result_df[col] = result_df[col].fillna(result_df[col].median())

        self.logger.info(f"Додано {added_features_count} ознак експоненціально зваженого вікна")

        return result_df
    def create_datetime_features(self, data: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:

        self.logger.info("Створення ознак на основі дати і часу...")

        # Перевірити, що індекс є DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є DatetimeIndex. Часові ознаки не будуть створені.")
            return data

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Лічильник доданих ознак
        added_features_count = 0

        # Створити базові ознаки
        # Година дня (0-23)
        result_df['hour'] = result_df.index.hour
        added_features_count += 1

        # День тижня (0-6, де 0 - понеділок, 6 - неділя)
        result_df['day_of_week'] = result_df.index.dayofweek
        added_features_count += 1

        # Номер дня місяця (1-31)
        result_df['day_of_month'] = result_df.index.day
        added_features_count += 1

        # Номер місяця (1-12)
        result_df['month'] = result_df.index.month
        added_features_count += 1

        # Номер кварталу (1-4)
        result_df['quarter'] = result_df.index.quarter
        added_features_count += 1

        # Номер року
        result_df['year'] = result_df.index.year
        added_features_count += 1

        # День року (1-366)
        result_df['day_of_year'] = result_df.index.dayofyear
        added_features_count += 1

        # Час доби (ранок, день, вечір, ніч)
        result_df['time_of_day'] = pd.cut(
            result_df.index.hour,
            bins=[-1, 5, 11, 17, 23],
            labels=['night', 'morning', 'day', 'evening']
        ).astype(str)
        added_features_count += 1

        # Чи це вихідний день (субота або неділя)
        result_df['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
        added_features_count += 1

        # Чи це останній день місяця
        result_df['is_month_end'] = result_df.index.is_month_end.astype(int)
        added_features_count += 1

        # Чи це останній день кварталу
        result_df['is_quarter_end'] = result_df.index.is_quarter_end.astype(int)
        added_features_count += 1

        # Чи це останній день року
        result_df['is_year_end'] = result_df.index.is_year_end.astype(int)
        added_features_count += 1

        # Якщо потрібні циклічні ознаки
        if cyclical:
            # Функція для створення циклічних ознак
            def create_cyclical_features(df, col, max_val):
                # Перетворення у радіани
                df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
                df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
                return df, 2  # Повертаємо df і кількість доданих ознак

            # Година (0-23)
            result_df, count = create_cyclical_features(result_df, 'hour', 24)
            added_features_count += count

            # День тижня (0-6)
            result_df, count = create_cyclical_features(result_df, 'day_of_week', 7)
            added_features_count += count

            # День місяця (1-31)
            result_df, count = create_cyclical_features(result_df, 'day_of_month', 31)
            added_features_count += count

            # Місяць (1-12)
            result_df, count = create_cyclical_features(result_df, 'month', 12)
            added_features_count += count

            # День року (1-366)
            result_df, count = create_cyclical_features(result_df, 'day_of_year', 366)
            added_features_count += count

        # Додаткові ознаки для криптовалютного ринку

        # Час доби за UTC (важливо для глобальних криптовалютних ринків)
        if 'hour' in result_df.columns:
            # Сегменти дня за UTC для виявлення патернів активності на різних ринках
            result_df['utc_segment'] = pd.cut(
                result_df['hour'],
                bins=[-1, 2, 8, 14, 20, 23],
                labels=['asia_late', 'asia_main', 'europe_main', 'america_main', 'asia_early']
            ).astype(str)
            added_features_count += 1

        # Опціонально: додавання відміток часу в Unix форматі (для розрахунків темпоральних відмінностей)
        result_df['timestamp'] = result_df.index.astype(np.int64) // 10 ** 9
        added_features_count += 1

        self.logger.info(f"Створено {added_features_count} часових ознак.")
        return result_df