from typing import List,  Tuple

import numpy as np
import pandas as pd

from featureengineering.feature_engineering import FeatureEngineering


class CrossFeatures(FeatureEngineering):
    """Клас для створення перехресних ознак із наявних числових стовпців."""

    def __init__(self):
        super().__init__()
        self.added_features = []  # Відстеження доданих ознак

    def _validate_columns(self, data: pd.DataFrame,
                          column_lists: List[List[str]],
                          list_names: List[str]) -> List[List[str]]:
        """Перевіряє наявність стовпців та повертає тільки дійсні стовпці.

        Args:
            data: Вхідний DataFrame
            column_lists: Список списків імен стовпців для перевірки
            list_names: Відповідні назви списків стовпців для логування

        Returns:
            Відфільтровані списки стовпців
        """
        validated_lists = []

        for columns, name in zip(column_lists, list_names):
            missing_cols = [col for col in columns if col not in data.columns]

            if missing_cols:
                self.logger.warning(f"{name} {missing_cols} не знайдено в даних і будуть пропущені")
                valid_cols = [col for col in columns if col in data.columns]
            else:
                valid_cols = columns.copy()

            validated_lists.append(valid_cols)

        return validated_lists

    def _handle_nan_values(self, df: pd.DataFrame, column: str, is_binary: bool = False) -> None:
        """Обробляє NaN значення у стовпці.

        Args:
            df: DataFrame для обробки
            column: Назва стовпця для обробки
            is_binary: Чи є стовпець бінарним
        """
        if not df[column].isna().any():
            return

        self.logger.debug(f"Заповнення NaN значень у стовпці {column}")

        if is_binary:
            df[column] = df[column].fillna(0)
        else:
            # Спочатку спробувати заповнити прямим і зворотнім методом
            df[column] = df[column].fillna(method='ffill').fillna(method='bfill')

            # Якщо все ще є NaN, заповнюємо нулями
            if df[column].isna().any():
                df[column] = df[column].fillna(0)

    def create_ratio_features(self, data: pd.DataFrame,
                              numerators: List[str],
                              denominators: List[str],
                              clip_percentiles: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
        """Створює ознаки-співвідношення між числовими стовпцями.

        Args:
            data: Вхідний DataFrame
            numerators: Список стовпців для використання у чисельнику
            denominators: Список стовпців для використання у знаменнику
            clip_percentiles: Перцентилі для обмеження значень (нижній, верхній)

        Returns:
            DataFrame з доданими ознаками співвідношень
        """
        self.logger.info("Створення ознак-співвідношень...")

        # Створюємо копію даних
        result_df = data.copy()

        # Перевіряємо наявність стовпців
        num_cols, den_cols = self._validate_columns(
            result_df, [numerators, denominators], ["Стовпці чисельника", "Стовпці знаменника"]
        )

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not num_cols or not den_cols:
            self.logger.error("Немає доступних стовпців для створення співвідношень")
            return result_df

        # Створюємо всі можливі комбінації співвідношень
        added_count = 0

        for num_col in num_cols:
            for den_col in den_cols:
                # Пропускаємо однакові стовпці
                if num_col == den_col:
                    self.logger.debug(f"Пропускаємо співвідношення {num_col}/{den_col} (однакові стовпці)")
                    continue

                # Створюємо назву нової ознаки
                ratio_name = f"ratio_{num_col}_to_{den_col}"

                # Перевіряємо наявність нульових значень в знаменнику
                zero_count = (result_df[den_col] == 0).sum()

                # Створюємо ознаку співвідношення
                if zero_count > 0:
                    self.logger.warning(f"Знаменник {den_col} містить {zero_count} нульових значень")
                    non_zero_mask = (result_df[den_col] != 0)

                    # Векторизоване обчислення для ненульових знаменників
                    result_df[ratio_name] = np.nan
                    result_df.loc[non_zero_mask, ratio_name] = (
                            result_df.loc[non_zero_mask, num_col] / result_df.loc[non_zero_mask, den_col]
                    )
                else:
                    result_df[ratio_name] = result_df[num_col] / result_df[den_col]

                # Обробляємо нескінченності
                inf_count = np.isinf(result_df[ratio_name]).sum()
                if inf_count > 0:
                    self.logger.warning(f"Співвідношення {ratio_name} містить {inf_count} нескінченних значень")
                    result_df[ratio_name].replace([np.inf, -np.inf], np.nan, inplace=True)

                # Заповнюємо NaN значення
                if result_df[ratio_name].isna().any():
                    median_val = result_df[ratio_name].median()
                    result_df[ratio_name] = result_df[ratio_name].fillna(
                        0 if pd.isna(median_val) else median_val
                    )

                # Застосовуємо обмеження на основі перцентилів
                q_low, q_high = result_df[ratio_name].quantile(list(clip_percentiles))
                result_df[ratio_name] = result_df[ratio_name].clip(q_low, q_high)

                self.added_features.append(ratio_name)
                added_count += 1

        self.logger.info(f"Додано {added_count} ознак-співвідношень")

        return result_df

    def create_crossover_features(self, data: pd.DataFrame,
                                  fast_columns: List[str],
                                  slow_columns: List[str],
                                  slope_periods: int = 3) -> pd.DataFrame:
        """Створює ознаки перетинів технічних індикаторів.

        Args:
            data: Вхідний DataFrame
            fast_columns: Список швидких індикаторів
            slow_columns: Список повільних індикаторів
            slope_periods: Кількість періодів для обчислення нахилу

        Returns:
            DataFrame з доданими ознаками перетинів
        """
        self.logger.info("Створення ознак перетинів індикаторів...")

        # Створюємо копію даних
        result_df = data.copy()

        # Перевіряємо наявність стовпців
        fast_cols, slow_cols = self._validate_columns(
            result_df, [fast_columns, slow_columns], ["Швидкі індикатори", "Повільні індикатори"]
        )

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not fast_cols or not slow_cols:
            self.logger.error("Немає доступних індикаторів для створення перетинів")
            return result_df

        added_count = 0

        # Векторизоване створення ознак для всіх пар індикаторів
        for fast_col in fast_cols:
            for slow_col in slow_cols:
                # Пропускаємо однакові індикатори
                if fast_col == slow_col:
                    self.logger.debug(f"Пропускаємо пару {fast_col}/{slow_col} (однакові індикатори)")
                    continue

                # Базова назва для ознак цієї пари
                base_name = f"{fast_col}_x_{slow_col}"

                # 1. Різниця між індикаторами (vectorized)
                diff_name = f"{base_name}_diff"
                result_df[diff_name] = result_df[fast_col] - result_df[slow_col]
                self.added_features.append(diff_name)
                added_count += 1

                # 2. Відносна різниця (у відсотках)
                rel_diff_name = f"{base_name}_rel_diff"
                non_zero_mask = (result_df[slow_col] != 0)

                # Векторизоване обчислення для ненульових знаменників
                result_df[rel_diff_name] = np.nan
                result_df.loc[non_zero_mask, rel_diff_name] = (
                        (result_df.loc[non_zero_mask, fast_col] / result_df.loc[non_zero_mask, slow_col] - 1) * 100
                )
                # Заповнюємо NaN значення
                result_df[rel_diff_name].fillna(0, inplace=True)
                self.added_features.append(rel_diff_name)
                added_count += 1

                # 3. Бінарні ознаки перетинів (vectorized)
                prev_diff = result_df[diff_name].shift(1)

                # Golden Cross: швидкий перетинає повільний знизу вгору
                golden_cross_name = f"{base_name}_golden_cross"
                result_df[golden_cross_name] = ((result_df[diff_name] > 0) & (prev_diff <= 0)).astype(int)
                self.added_features.append(golden_cross_name)
                added_count += 1

                # Death Cross: швидкий перетинає повільний згори вниз
                death_cross_name = f"{base_name}_death_cross"
                result_df[death_cross_name] = ((result_df[diff_name] < 0) & (prev_diff >= 0)).astype(int)
                self.added_features.append(death_cross_name)
                added_count += 1

                # 4. Напрямок (vectorized)
                direction_name = f"{base_name}_direction"
                result_df[direction_name] = np.sign(result_df[diff_name]).fillna(0).astype(int)
                self.added_features.append(direction_name)
                added_count += 1

                # 5. Різниця нахилів індикаторів (vectorized)
                slope_name = f"{base_name}_slope_diff"
                result_df[slope_name] = (
                        result_df[fast_col].diff(slope_periods) - result_df[slow_col].diff(slope_periods)
                )
                self.added_features.append(slope_name)
                added_count += 1

                # 6. Оптимізована тривалість стану
                self._add_state_duration_feature(result_df, base_name, golden_cross_name, death_cross_name)
                self.added_features.append(f"{base_name}_state_duration")
                added_count += 1

        # Заповнюємо NaN для всіх нових ознак
        for col in self.added_features:
            if col not in data.columns:  # Перевіряємо, що це нова ознака
                self._handle_nan_values(result_df, col, is_binary=col.endswith('_cross'))

        self.logger.info(f"Додано {added_count} ознак перетинів індикаторів")

        return result_df

    def _add_state_duration_feature(self, df: pd.DataFrame, base_name: str,
                                    golden_cross_name: str, death_cross_name: str) -> None:
        """Додає ознаку тривалості поточного стану (оптимізована версія).

        Args:
            df: DataFrame для обробки
            base_name: Базова назва для ознаки
            golden_cross_name: Назва стовпця golden cross
            death_cross_name: Назва стовпця death cross
        """
        duration_name = f"{base_name}_state_duration"

        # Знаходимо всі перетини
        all_crosses = (df[golden_cross_name] == 1) | (df[death_cross_name] == 1)

        # Використовуємо numpy для ефективного обчислення тривалості стану
        # Створюємо масив з кумулятивним числом без перетину
        cross_indices = np.where(all_crosses)[0]
        duration = np.zeros(len(df))

        if len(cross_indices) > 0:
            # Використовуємо numpy масиви для швидкої операції
            segment_starts = np.zeros(len(df), dtype=int)

            # Позначаємо початки сегментів
            segment_starts[cross_indices] = 1

            # Для кожної точки знаходимо відстань до останнього перетину
            # використовуючи кумулятивну суму як лічильник з моменту перетину
            cumulative_idx = np.cumsum(np.arange(len(df)))
            last_cross_cumidx = np.maximum.accumulate(cumulative_idx * segment_starts)

            # Обчислюємо тривалість станів
            duration = np.where(
                segment_starts == 1,
                0,  # На точках перетину тривалість = 0
                cumulative_idx - last_cross_cumidx  # Інакше різниця між поточним та останнім перетином
            )

        df[duration_name] = duration