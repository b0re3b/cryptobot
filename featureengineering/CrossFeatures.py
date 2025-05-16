from typing import List

import numpy as np
import pandas as pd

from featureengineering.feature_engineering import FeatureEngineering


class CrossFeatures(FeatureEngineering):

    def create_ratio_features(self, data: pd.DataFrame,
                              numerators: List[str],
                              denominators: List[str]) -> pd.DataFrame:

        self.logger.info("Створення ознак-співвідношень...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що всі зазначені стовпці існують в даних
        missing_numerators = [col for col in numerators if col not in result_df.columns]
        missing_denominators = [col for col in denominators if col not in result_df.columns]

        if missing_numerators:
            self.logger.warning(f"Стовпці чисельника {missing_numerators} не знайдено в даних і будуть пропущені")
            numerators = [col for col in numerators if col in result_df.columns]

        if missing_denominators:
            self.logger.warning(f"Стовпці знаменника {missing_denominators} не знайдено в даних і будуть пропущені")
            denominators = [col for col in denominators if col in result_df.columns]

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not numerators or not denominators:
            self.logger.error("Немає доступних стовпців для створення співвідношень")
            return result_df

        # Лічильник доданих ознак
        added_features_count = 0

        # Створюємо всі можливі комбінації співвідношень
        for num_col in numerators:
            for den_col in denominators:
                # Пропускаємо деякі комбінації, якщо чисельник і знаменник однакові
                if num_col == den_col:
                    self.logger.debug(f"Пропускаємо співвідношення {num_col}/{den_col} (однакові стовпці)")
                    continue

                # Створюємо назву нової ознаки
                ratio_name = f"ratio_{num_col}_to_{den_col}"

                # Обробляємо випадки з нульовими знаменниками
                # Використовуємо numpy.divide з параметром where для безпечного ділення
                self.logger.debug(f"Створюємо співвідношення {ratio_name}")

                # Перевіряємо, чи є нульові значення в знаменнику
                zero_denominator_count = (result_df[den_col] == 0).sum()
                if zero_denominator_count > 0:
                    self.logger.warning(f"Знаменник {den_col} містить {zero_denominator_count} нульових значень")

                    # Використовуємо безпечне ділення: ігноруємо ділення на нуль
                    # і встановлюємо спеціальне значення для таких випадків
                    denominator = result_df[den_col].copy()

                    # Створюємо маску для ненульових значень
                    non_zero_mask = (denominator != 0)

                    # Виконуємо ділення тільки для ненульових знаменників
                    result_df[ratio_name] = np.nan  # спочатку встановлюємо NaN
                    result_df.loc[non_zero_mask, ratio_name] = result_df.loc[non_zero_mask, num_col] / denominator[
                        non_zero_mask]

                    # Для нульових знаменників можна встановити спеціальне значення або залишити NaN
                    # Тут ми залишаємо NaN і потім заповнюємо їх
                else:
                    # Якщо нульових знаменників немає, просто ділимо
                    result_df[ratio_name] = result_df[num_col] / result_df[den_col]

                # Обробляємо випадки з нескінченностями (можуть виникнути при діленні на дуже малі числа)
                inf_count = np.isinf(result_df[ratio_name]).sum()
                if inf_count > 0:
                    self.logger.warning(f"Співвідношення {ratio_name} містить {inf_count} нескінченних значень")
                    # Замінюємо нескінченності на NaN для подальшої обробки
                    result_df[ratio_name].replace([np.inf, -np.inf], np.nan, inplace=True)

                # Заповнюємо NaN значення (якщо є)
                if result_df[ratio_name].isna().any():
                    # Заповнюємо NaN медіаною стовпця
                    median_val = result_df[ratio_name].median()
                    if pd.isna(median_val):  # Якщо медіана теж NaN
                        result_df[ratio_name] = result_df[ratio_name].fillna(0)
                        self.logger.debug(f"Заповнення NaN значень у стовпці {ratio_name} нулями")
                    else:
                        result_df[ratio_name] = result_df[ratio_name].fillna(median_val)
                        self.logger.debug(f"Заповнення NaN значень у стовпці {ratio_name} медіаною: {median_val}")

                # Додаємо опціональне обмеження на великі значення
                # Можна використовувати вінсоризацію або кліпінг
                # Тут використовуємо простий кліпінг на основі перцентилів
                q_low, q_high = result_df[ratio_name].quantile([0.01, 0.99])
                result_df[ratio_name] = result_df[ratio_name].clip(q_low, q_high)

                added_features_count += 1

        self.logger.info(f"Додано {added_features_count} ознак-співвідношень")

        return result_df
    def create_crossover_features(self, data: pd.DataFrame,
                                  fast_columns: List[str],
                                  slow_columns: List[str]) -> pd.DataFrame:

        self.logger.info("Створення ознак перетинів індикаторів...")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що всі зазначені стовпці існують в даних
        missing_fast = [col for col in fast_columns if col not in result_df.columns]
        missing_slow = [col for col in slow_columns if col not in result_df.columns]

        if missing_fast:
            self.logger.warning(f"Швидкі індикатори {missing_fast} не знайдено в даних і будуть пропущені")
            fast_columns = [col for col in fast_columns if col in result_df.columns]

        if missing_slow:
            self.logger.warning(f"Повільні індикатори {missing_slow} не знайдено в даних і будуть пропущені")
            slow_columns = [col for col in slow_columns if col in result_df.columns]

        # Перевіряємо, чи залишились стовпці після фільтрації
        if not fast_columns or not slow_columns:
            self.logger.error("Немає доступних індикаторів для створення перетинів")
            return result_df

        # Лічильник доданих ознак
        added_features_count = 0

        # Для кожної пари індикаторів створюємо ознаки перетинів
        for fast_col in fast_columns:
            for slow_col in slow_columns:
                # Пропускаємо пари однакових індикаторів
                if fast_col == slow_col:
                    self.logger.debug(f"Пропускаємо пару {fast_col}/{slow_col} (однакові індикатори)")
                    continue

                # Базова назва для ознак цієї пари
                base_name = f"{fast_col}_x_{slow_col}"

                # 1. Створюємо ознаку відносної різниці між індикаторами
                diff_name = f"{base_name}_diff"
                result_df[diff_name] = result_df[fast_col] - result_df[slow_col]
                added_features_count += 1

                # 2. Створюємо ознаку відносної різниці (у відсотках)
                rel_diff_name = f"{base_name}_rel_diff"
                # Уникаємо ділення на нуль
                non_zero_mask = (result_df[slow_col] != 0)
                result_df[rel_diff_name] = np.nan
                result_df.loc[non_zero_mask, rel_diff_name] = (
                        (result_df.loc[non_zero_mask, fast_col] / result_df.loc[non_zero_mask, slow_col] - 1) * 100
                )
                # Заповнюємо NaN значення
                result_df[rel_diff_name].fillna(0, inplace=True)
                added_features_count += 1

                # 3. Створюємо бінарні ознаки перетинів
                # Визначаємо попередні значення різниці для виявлення перетинів
                prev_diff = result_df[diff_name].shift(1)

                # Golden Cross: швидкий індикатор перетинає повільний знизу вгору
                golden_cross_name = f"{base_name}_golden_cross"
                result_df[golden_cross_name] = ((result_df[diff_name] > 0) & (prev_diff <= 0)).astype(int)
                added_features_count += 1

                # Death Cross: швидкий індикатор перетинає повільний згори вниз
                death_cross_name = f"{base_name}_death_cross"
                result_df[death_cross_name] = ((result_df[diff_name] < 0) & (prev_diff >= 0)).astype(int)
                added_features_count += 1

                # 4. Створюємо ознаку тривалості поточного стану (кількість періодів після останнього перетину)
                duration_name = f"{base_name}_state_duration"

                # Ініціалізуємо значення тривалості
                result_df[duration_name] = 0

                # Знаходимо індекси всіх перетинів (обох типів)
                all_crosses = (result_df[golden_cross_name] == 1) | (result_df[death_cross_name] == 1)
                cross_indices = np.where(all_crosses)[0]

                if len(cross_indices) > 0:
                    # Для кожного сегмента між перетинами встановлюємо тривалість
                    prev_idx = 0
                    for idx in cross_indices:
                        if idx > 0:  # Пропускаємо перший перетин (немає даних до нього)
                            # Збільшуємо тривалість для всіх точок у сегменті
                            for i in range(prev_idx, idx):
                                result_df.iloc[i, result_df.columns.get_loc(duration_name)] = i - prev_idx
                        prev_idx = idx

                    # Обробляємо останній сегмент до кінця даних
                    for i in range(prev_idx, len(result_df)):
                        result_df.iloc[i, result_df.columns.get_loc(duration_name)] = i - prev_idx

                added_features_count += 1

                # 5. Додаємо ознаку напрямку (1 якщо швидкий вище повільного, -1 якщо нижче)
                direction_name = f"{base_name}_direction"
                result_df[direction_name] = np.sign(result_df[diff_name]).fillna(0).astype(int)
                added_features_count += 1

                # 6. Додаємо ознаку кутового коефіцієнта (нахилу) між індикаторами
                # Обчислюємо різницю похідних індикаторів для оцінки відносної швидкості зміни
                slope_name = f"{base_name}_slope_diff"
                # Використовуємо різницю за 3 періоди для стабільнішого результату
                fast_slope = result_df[fast_col].diff(3)
                slow_slope = result_df[slow_col].diff(3)
                result_df[slope_name] = fast_slope - slow_slope
                added_features_count += 1

        # Заповнюємо NaN значення для нових ознак
        for col in result_df.columns:
            if col not in data.columns:  # Перевіряємо, що це нова ознака
                if result_df[col].isna().any():
                    self.logger.debug(f"Заповнення NaN значень у стовпці {col}")
                    # Для бінарних ознак (перетини) використовуємо 0
                    if col.endswith('_cross'):
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        # Для інших ознак використовуємо прямий і зворотній метод заповнення
                        result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

                        # Якщо все ще є NaN, заповнюємо нулями
                        if result_df[col].isna().any():
                            result_df[col] = result_df[col].fillna(0)

        self.logger.info(f"Додано {added_features_count} ознак перетинів індикаторів")

        return result_df