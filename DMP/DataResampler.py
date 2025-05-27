import json
from datetime import datetime

import numpy as np
import pandas as pd
import dask.dataframe as dd
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
from functools import lru_cache

class DataResampler:
    def __init__(self, logger, chunk_size=500_000, scaling_sample_size= 1_000_000):
        self.logger = logger
        self.scalers = {}
        self.original_data_map = {}
        self.chunk_size = chunk_size
        self.scaling_sample_size = scaling_sample_size
        self.find_column = self._find_column_original
        self.cache = {}

    def _find_column_original(self, df, column_name):
        """Знаходить колонку незалежно від регістру з обробкою конфліктів"""
        exact_match = [col for col in df.columns if col == column_name]
        if exact_match:
            return exact_match[0]

        case_insensitive_matches = [col for col in df.columns if col.lower() == column_name.lower()]
        if case_insensitive_matches:
            if len(case_insensitive_matches) > 1:
                self.logger.warning(
                    f"Знайдено кілька варіантів для колонки '{column_name}': {case_insensitive_matches}. Використовуємо перший.")
            return case_insensitive_matches[0]

        return None

    def detect_interval(self, data: pd.DataFrame) -> str:

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("DataFrame повинен мати DatetimeIndex для визначення інтервалу")
            return None

        if len(data) < 2:
            self.logger.error("Потрібно щонайменше 2 точки даних для визначення інтервалу")
            return None

        # Обчислюємо різницю в часі між сусідніми індексами
        time_diffs = data.index.to_series().diff().dropna()

        # Визначаємо найбільш поширену різницю (моду)
        from collections import Counter
        counter = Counter(time_diffs)
        most_common_diff = counter.most_common(1)[0][0]

        # Перетворюємо в загальні одиниці виміру (секунди)
        diff_seconds = most_common_diff.total_seconds()

        # Визначаємо інтервал на основі кількості секунд
        if diff_seconds < 60:
            interval = f"{int(diff_seconds)}s"
        elif diff_seconds < 3600:
            minutes = int(diff_seconds / 60)
            interval = f"{minutes}m"
        elif diff_seconds < 86400:
            hours = int(diff_seconds / 3600)
            interval = f"{hours}h"
        elif diff_seconds < 604800:
            days = int(diff_seconds / 86400)
            interval = f"{days}d"
        elif diff_seconds < 2592000:
            weeks = int(diff_seconds / 604800)
            interval = f"{weeks}w"
        else:
            months = max(1, int(diff_seconds / 2592000))
            interval = f"{months}M"

        self.logger.info(f"Визначений інтервал даних: {interval}")

        # Зберігаємо результат для подальшого використання
        self.original_data_map['detected_interval'] = interval
        return interval

    def auto_resample(self, data: pd.DataFrame, target_interval: str = None,
                      scaling_factor: int = None) -> pd.DataFrame:

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("DataFrame повинен мати DatetimeIndex для ресемплінгу")
            return data

        # Визначаємо поточний інтервал
        current_interval = self.detect_interval(data)
        if not current_interval:
            self.logger.error("Не вдалося визначити поточний інтервал даних")
            return data

        # Якщо цільовий інтервал не задано, використовуємо scaling_factor
        if target_interval is None:
            if scaling_factor is None:
                # За замовчуванням множимо інтервал на 4
                scaling_factor = 4
                self.logger.info(f"Використовується масштабуючий коефіцієнт за замовчуванням: {scaling_factor}")

            import re
            match = re.match(r'(\d+)([smhdwM])', current_interval)
            if not match:
                self.logger.error(f"Неправильний формат поточного інтервалу: {current_interval}")
                return data

            number, unit = match.groups()
            number = int(number) * scaling_factor
            target_interval = f"{number}{unit}"
            self.logger.info(f"Обчислений цільовий інтервал: {target_interval}")

        # Здійснюємо ресемплінг
        self.logger.info(f"Виконується зміна інтервалу: {current_interval} -> {target_interval}")
        resampled_data = self.resample_data(data, target_interval)

        # Зберігаємо інформацію про трансформацію
        self.original_data_map['auto_resample_info'] = {
            'original_interval': current_interval,
            'target_interval': target_interval,
            'scaling_factor': scaling_factor,
            'original_shape': data.shape,
            'resampled_shape': resampled_data.shape
        }

        return resampled_data

    def suggest_intervals(self, data: pd.DataFrame, max_suggestions: int = 5) -> list:

        current_interval = self.detect_interval(data)
        if not current_interval:
            return []

        import re
        match = re.match(r'(\d+)([smhdwM])', current_interval)
        if not match:
            return []

        number, unit = match.groups()
        number = int(number)

        # Стандартні множники для різних одиниць
        standard_multipliers = {
            's': [5, 10, 15, 30, 60],
            'm': [5, 10, 15, 30, 60],
            'h': [2, 3, 4, 6, 8, 12, 24],
            'd': [2, 3, 5, 7, 10, 14, 30],
            'w': [2, 3, 4],
            'M': [2, 3, 6, 12]
        }

        multipliers = standard_multipliers.get(unit, [2, 3, 4, 5])

        # Формуємо список рекомендацій
        suggestions = []
        for multiplier in multipliers:
            new_value = number * multiplier

            if unit == 's' and new_value >= 60:
                suggestions.append(f"{new_value // 60}m")
            elif unit == 'm' and new_value >= 60:
                suggestions.append(f"{new_value // 60}h")
            elif unit == 'h' and new_value >= 24:
                suggestions.append(f"{new_value // 24}d")
            elif unit == 'd' and new_value >= 7 and new_value % 7 == 0:
                suggestions.append(f"{new_value // 7}w")
            elif unit == 'd' and new_value >= 30:
                suggestions.append(f"{new_value // 30}M")
            else:
                suggestions.append(f"{new_value}{unit}")

        return suggestions[:max_suggestions]

    def _optimize_aggregation_dict(self, data: pd.DataFrame, store_column_map: bool = False) -> Dict:
        """
        Формує словник агрегацій на основі типу колонок і їх назв (уніфікована логіка).
        """
        agg_dict = {}
        columns_lower_map = {col.lower(): col for col in data.columns}
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Основні фінансові колонки
        standard_aggs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'trades': 'sum',
            'taker_buy_volume': 'sum',
            'taker_sell_volume': 'sum',
            'taker_buy_base_volume': 'sum',
            'taker_buy_quote_volume': 'sum',
            'quote_volume': 'sum',
            'quote_asset_volume': 'sum',
            'number_of_trades': 'sum',
            'vwap': 'mean',
            'funding_rate': 'mean',
        }

        for base_col_lower, agg_method in standard_aggs.items():
            if base_col_lower in columns_lower_map:
                actual_col = columns_lower_map[base_col_lower]
                agg_dict[actual_col] = agg_method
                if store_column_map:
                    self.original_data_map[f"{base_col_lower}_column"] = actual_col

        # Обробка решти числових колонок
        for col in numeric_cols:
            if col not in agg_dict:
                col_lower = col.lower()
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount', 'volume']):
                    agg_dict[col] = 'sum'
                elif any(x in col_lower for x in ['id', 'code', 'identifier']):
                    agg_dict[col] = 'last'
                elif any(x in col_lower for x in ['price', 'rate', 'fee', 'vwap']):
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'mean'

        # Обробка нечислових колонок
        non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = 'last'

        return agg_dict

    def convert_interval_to_pandas_format(self, timeframe: str) -> str:

        if not timeframe or not isinstance(timeframe, str):
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        interval_map = {
            's': 'S',  # секунди
            'm': 'T',  # хвилини (в pandas використовується 'T' для хвилин)
            'h': 'H',  # години
            'd': 'D',  # дні
            'w': 'W',  # тижні
            'M': 'M',  # місяці
        }

        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        # Перевірка, чи число додатне
        if int(number) <= 0:
            raise ValueError(f"Інтервал повинен бути додатнім: {timeframe}")

        # Перевірка, чи підтримується одиниця часу
        if unit not in interval_map:
            raise ValueError(f"Непідтримувана одиниця часу: {unit}")

        # Конвертуємо у pandas формат
        pandas_interval = f"{number}{interval_map[unit]}"
        self.logger.info(f"Перетворено інтервал '{timeframe}' у pandas формат '{pandas_interval}'")

        return pandas_interval

    def resample_data(self, data: pd.DataFrame, target_interval: str,
                      required_columns: List[str] = None,
                      auto_detect: bool = True,
                      check_interval_compatibility: bool = True) -> pd.DataFrame:

        self.logger.info(f"Початок ресемплінгу даних. Наявні колонки: {list(data.columns)}")

        if data.empty:
            self.logger.warning("Отримано порожній DataFrame для ресемплінгу")
            return data

        # ДОДАНО: Детальна перевірка та корекція індексу
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертації...")
                data = data.copy()
                data.index = pd.to_datetime(data.index)
                self.logger.info("Індекс успішно конвертовано у DatetimeIndex")
            except Exception as e:
                self.logger.error(f"Не вдалося конвертувати індекс у DatetimeIndex: {str(e)}")
                raise ValueError("Дані повинні мати DatetimeIndex для ресемплінгу")

        # ДОДАНО: Перевірка та корекція порядку індексу
        if not data.index.is_monotonic_increasing:
            self.logger.warning("Індекс не відсортований. Сортуємо дані за часовим індексом.")
            data = data.sort_index()

        # ДОДАНО: Перевірка дублікатів у індексі
        duplicated_indices = data.index.duplicated()
        if duplicated_indices.any():
            dup_count = duplicated_indices.sum()
            self.logger.warning(f"Виявлено {dup_count} дублікатів у індексі. Видаляємо дублікати...")
            data = data[~duplicated_indices]

        # Збереження списку оригінальних колонок
        original_columns = set(data.columns)
        self.logger.info(f"Початкові колонки: {original_columns}")

        # Зберігаємо оригінальні дані
        self.original_data_map['original_data'] = data.copy()

        # Записуємо початковий інтервал для перевірки
        initial_index_diff = None
        initial_intervals = []
        if len(data) > 10:  # Беремо кілька перших різниць для більшої надійності
            for i in range(1, min(10, len(data))):
                diff_seconds = (data.index[i] - data.index[i - 1]).total_seconds()
                initial_intervals.append(diff_seconds)

            initial_index_diff = sum(initial_intervals) / len(initial_intervals)
            self.logger.info(f"Початковий середній інтервал в секундах: {initial_index_diff}")
            self.logger.info(f"Перші кілька інтервалів: {initial_intervals}")
        elif len(data) > 1:
            initial_index_diff = (data.index[1] - data.index[0]).total_seconds()
            self.logger.info(f"Початковий інтервал в секундах: {initial_index_diff}")

        # Автоматичне визначення поточного інтервалу, якщо потрібно
        current_interval = None
        if auto_detect:
            current_interval = self.detect_interval(data)
            if not current_interval:
                self.logger.warning("Не вдалося визначити поточний інтервал даних, продовжуємо без перевірок")
            else:
                self.logger.info(f"Визначено поточний інтервал даних: {current_interval}")

                # Перевірка сумісності інтервалів, якщо потрібно
                if check_interval_compatibility and current_interval:
                    try:
                        current_timedelta = self.parse_interval(current_interval)
                        target_timedelta = self.parse_interval(target_interval)

                        if target_timedelta < current_timedelta:
                            self.logger.warning(
                                f"Цільовий інтервал '{target_interval}' менший за поточний '{current_interval}'. "
                                f"Ресемплінг до менших інтервалів може призвести до втрати інформації."
                            )
                    except Exception as e:
                        self.logger.error(f"Помилка при перевірці сумісності інтервалів: {str(e)}")

        # Перевірка необхідних колонок
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Перевірка наявності колонок (незалежно від регістру)
        data_columns_lower = {col.lower(): col for col in data.columns}
        missing_cols = [
            col for col in required_columns
            if col.lower() not in data_columns_lower
        ]

        if missing_cols:
            self.logger.warning(f"Відсутні необхідні колонки: {missing_cols}")
            if len(missing_cols) == len(required_columns):
                self.logger.error("Неможливо виконати ресемплінг без необхідних колонок даних")
                return data
            else:
                self.logger.warning("Ресемплінг буде виконано, але результати можуть бути неповними")

        # Перевірка інтервалу часу
        try:
            pandas_interval = self.convert_interval_to_pandas_format(target_interval)
            self.logger.info(f"Ресемплінг даних до інтервалу: {target_interval} (pandas формат: {pandas_interval})")

            # ДОДАНО: Обчислення очікуваного інтервалу в секундах для подальшої перевірки
            expected_interval_td = self.parse_interval(target_interval)
            expected_interval_seconds = expected_interval_td.total_seconds()
            self.logger.info(f"Очікуваний інтервал ресемплінгу: {expected_interval_seconds} сек.")
        except ValueError as e:
            self.logger.error(f"Неправильний формат інтервалу: {str(e)}")
            return data

        # ДОДАНО: Перевірка нульового інтервалу
        if expected_interval_seconds <= 0:
            self.logger.error(f"Очікуваний інтервал повинен бути додатнім: {expected_interval_seconds} сек.")
            return data

        # Оптимізована підготовка агрегацій
        agg_dict = self._optimize_aggregation_dict(data)

        # ДОДАНО: Логування методів агрегації для діагностики
        self.logger.info(f"Методи агрегації для колонок: {json.dumps({str(k): str(v) for k, v in agg_dict.items()})}")

        # Оптимізована обробка великих наборів даних
        batch_size = self.chunk_size  # Використовуємо налаштування класу
        total_rows = len(data)

        # ЗМІНЕНО: Розширені параметри для точного контролю над ресемплінгом
        resample_params = {
            'rule': pandas_interval,
            'closed': 'left',  # Включаємо лівий край інтервалу
            'label': 'left',  # Встановлюємо мітку на лівий край
            'origin': 'start',  # Початок відліку від першої точки даних
            'offset': '0s'  # Зсув на початку відліку
        }

        self.logger.info(f"Параметри ресемплінгу: {resample_params}")

        # ДОДАНО: Збереження початкового часового діапазону для перевірки
        orig_start = data.index.min()
        orig_end = data.index.max()
        self.logger.info(f"Діапазон вхідних даних: {orig_start} - {orig_end}")

        # Стратегія обробки залежно від розміру даних
        if total_rows <= batch_size:
            # Якщо дані менші за batch_size, обробляємо весь DataFrame
            try:
                # ЗМІНЕНО: використовуємо розширені параметри
                resampled = data.resample(**resample_params).agg(agg_dict)

                # ДОДАНО: перевірка на пустий результат
                if resampled.empty:
                    self.logger.error("Ресемплінг призвів до порожнього DataFrame! Повертаємо оригінальні дані.")
                    return data

                # Заповнення відсутніх значень з оптимізацією
                resampled = self._fill_missing_values(resampled)

                # ДОДАНО: детальніша перевірка результатів ресемплінгу
                if len(resampled) > 1:
                    # Перевіряємо кілька інтервалів, а не лише перший
                    interval_checks = []
                    for i in range(1, min(10, len(resampled))):
                        actual_interval = (resampled.index[i] - resampled.index[i - 1]).total_seconds()
                        interval_checks.append(actual_interval)

                    avg_interval = sum(interval_checks) / len(interval_checks)
                    self.logger.info(f"Перевірені інтервали після ресемплінгу: {interval_checks}")
                    self.logger.info(
                        f"Середній новий інтервал: {avg_interval}, очікуваний: {expected_interval_seconds}")

                    # Перевіряємо приблизну відповідність із допустимим відхиленням 5%
                    interval_ratio = abs(avg_interval / expected_interval_seconds - 1)

                    if interval_ratio > 0.05:  # 5% tolerance
                        self.logger.error(
                            f"УВАГА! Ресемплінг виконано з відхиленням від очікуваного інтервалу! "
                            f"Отриманий середній інтервал {avg_interval:.2f} сек. "
                            f"відрізняється від очікуваного {expected_interval_seconds:.2f} сек. на {interval_ratio:.2%}"
                        )
                    else:
                        self.logger.info(f"Ресемплінг успішно виконано з правильним інтервалом "
                                         f"(відхилення: {interval_ratio:.2%})")

                    # ДОДАНО: перевірка розміру даних
                    reduction_ratio = 1 - (len(resampled) / total_rows)
                    if reduction_ratio < 0.01 and total_rows > 100:  # Майже немає змін у розмірі
                        self.logger.warning(
                            f"Ресемплінг майже не змінив кількість рядків: {total_rows} -> {len(resampled)} "
                            f"(зменшення лише на {reduction_ratio:.2%}). Можливо, проблема з параметрами ресемплінгу."
                        )

                self.original_data_map['resampled_data'] = resampled.copy()

                # Зберігаємо інформацію про трансформацію
                self.original_data_map['resample_info'] = {
                    'original_interval': current_interval if auto_detect and current_interval else "unknown",
                    'target_interval': target_interval,
                    'original_shape': data.shape,
                    'resampled_shape': resampled.shape,
                    'initial_index_diff_seconds': initial_index_diff,
                    'new_index_diff_seconds': avg_interval if len(interval_checks) > 0 else None,
                    'reduction_ratio': reduction_ratio if len(resampled) > 0 else None
                }

                # ДОДАНО: додаткова перевірка часового діапазону
                resampled_start = resampled.index.min()
                resampled_end = resampled.index.max()
                self.logger.info(f"Діапазон вихідних даних: {resampled_start} - {resampled_end}")

                if resampled_start > orig_start or resampled_end < orig_end:
                    self.logger.warning(
                        f"Увага! Часовий діапазон змінився після ресемплінгу! "
                        f"Можливо втрачено дані на початку або в кінці."
                    )

                return resampled
            except Exception as e:
                self.logger.error(f"Помилка при ресемплінгу: {str(e)}")
                import traceback
                self.logger.error(f"Деталі помилки: {traceback.format_exc()}")
                return data

        # Batch-обробка для великих наборів даних
        self.logger.info(f"Застосовуємо батч-обробку для великого набору даних: {total_rows} рядків")
        result_batches = []
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = data.iloc[start:end]

            try:
                # Ресемплінг батчу з оновленими параметрами
                resampled_batch = batch.resample(**resample_params).agg(agg_dict)

                # Перевірка результату батчу
                if not resampled_batch.empty:
                    result_batches.append(resampled_batch)
                else:
                    self.logger.warning(f"Батч {start}-{end} дав порожній результат, пропускаємо")

            except Exception as e:
                self.logger.error(f"Помилка при обробці батчу {start}-{end}: {str(e)}")
                continue

        # Перевірка результатів
        if not result_batches:
            self.logger.error("Жоден батч не дав результатів. Повертаємо оригінальні дані.")
            return data

        # Об'єднання результатів
        try:
            # ЗМІНЕНО: використовуємо метод concat з правильними параметрами
            resampled = pd.concat(result_batches, ignore_index=False, verify_integrity=True)

            # ДОДАНО: перевірка і видалення можливих дублікатів після конкатенації
            if resampled.index.duplicated().any():
                self.logger.warning("Виявлено дублікати після об'єднання батчів, видаляємо...")
                resampled = resampled[~resampled.index.duplicated(keep='first')]

            # Заповнення відсутніх значень
            resampled = self._fill_missing_values(resampled)

            # ДОДАНО: перевірка успішності ресемплінгу для batch-обробки
            if len(resampled) > 1:
                interval_checks = []
                for i in range(1, min(10, len(resampled))):
                    actual_interval = (resampled.index[i] - resampled.index[i - 1]).total_seconds()
                    interval_checks.append(actual_interval)

                avg_interval = sum(interval_checks) / len(interval_checks)
                self.logger.info(f"Перевірені інтервали після batch-ресемплінгу: {interval_checks}")
                self.logger.info(f"Середній новий інтервал: {avg_interval}, очікуваний: {expected_interval_seconds}")

                # Перевіряємо приблизну відповідність
                interval_ratio = abs(avg_interval / expected_interval_seconds - 1)

                if interval_ratio > 0.05:  # 5% tolerance
                    self.logger.error(
                        f"УВАГА! Batch-ресемплінг виконано з відхиленням від очікуваного інтервалу! "
                        f"Отриманий середній інтервал {avg_interval:.2f} сек. "
                        f"відрізняється від очікуваного {expected_interval_seconds:.2f} сек. на {interval_ratio:.2%}"
                    )
                else:
                    self.logger.info(f"Batch-ресемплінг успішно виконано з правильним інтервалом "
                                     f"(відхилення: {interval_ratio:.2%})")

                # ДОДАНО: перевірка розміру даних
                reduction_ratio = 1 - (len(resampled) / total_rows)
                if reduction_ratio < 0.01 and total_rows > 100:  # Майже немає змін у розмірі
                    self.logger.warning(
                        f"Batch-ресемплінг майже не змінив кількість рядків: {total_rows} -> {len(resampled)} "
                        f"(зменшення лише на {reduction_ratio:.2%}). Можливо, проблема з параметрами ресемплінгу."
                    )

            self.original_data_map['resampled_data'] = resampled.copy()

            # Зберігаємо інформацію про трансформацію
            self.original_data_map['resample_info'] = {
                'original_interval': current_interval if auto_detect and current_interval else "unknown",
                'target_interval': target_interval,
                'original_shape': data.shape,
                'resampled_shape': resampled.shape,
                'batch_processing': True,
                'batches_count': len(result_batches),
                'initial_index_diff_seconds': initial_index_diff,
                'new_index_diff_seconds': avg_interval if 'avg_interval' in locals() else None,
                'reduction_ratio': reduction_ratio if 'reduction_ratio' in locals() else None
            }

            # ДОДАНО: додаткова перевірка часового діапазону при batch-обробці
            resampled_start = resampled.index.min()
            resampled_end = resampled.index.max()
            self.logger.info(f"Діапазон вихідних даних після batch-обробки: {resampled_start} - {resampled_end}")

            if resampled_start > orig_start or resampled_end < orig_end:
                self.logger.warning(
                    f"Увага! Часовий діапазон змінився після batch-ресемплінгу! "
                    f"Можливо втрачено дані на початку або в кінці."
                )

            self.logger.info(
                f"Ресемплінг успішно завершено: {resampled.shape[0]} рядків, {len(resampled.columns)} колонок")
            return resampled

        except Exception as e:
            self.logger.error(f"Помилка при об'єднанні батчів: {str(e)}")
            import traceback
            self.logger.error(f"Деталі помилки: {traceback.format_exc()}")
            return data

    def verify_resampling(self, original_data: pd.DataFrame, resampled_data: pd.DataFrame,
                          target_interval: str) -> bool:
        """Перевіряє, чи був ресемплінг виконаний правильно"""

        if len(resampled_data) <= 1:
            self.logger.warning("Недостатньо даних для перевірки ресемплінгу")
            return False

        # 1. Перевірка зміни кількості рядків
        orig_rows = len(original_data)
        new_rows = len(resampled_data)

        if orig_rows == new_rows:
            self.logger.warning(f"Кількість рядків не змінилася: {orig_rows}. Ресемплінг може бути неправильним.")
        else:
            self.logger.info(f"Зміна кількості рядків: {orig_rows} -> {new_rows}")

        # 2. Перевірка інтервалу
        expected_td = self.parse_interval(target_interval)
        expected_seconds = expected_td.total_seconds()

        actual_intervals = []
        for i in range(1, min(10, len(resampled_data))):
            actual_seconds = (resampled_data.index[i] - resampled_data.index[i - 1]).total_seconds()
            actual_intervals.append(actual_seconds)

        avg_actual = sum(actual_intervals) / len(actual_intervals)

        self.logger.info(f"Очікуваний інтервал: {expected_seconds} сек.")
        self.logger.info(f"Фактичний середній інтервал: {avg_actual} сек.")
        self.logger.info(f"Перші інтервали: {actual_intervals}")

        # Перевірка з допуском 5%
        is_correct = abs(avg_actual - expected_seconds) <= expected_seconds * 0.05

        if not is_correct:
            self.logger.error(f"Ресемплінг виконано неправильно! Різниця: {abs(avg_actual - expected_seconds)} сек.")
        else:
            self.logger.info("Ресемплінг виконано правильно!")

        return is_correct

    def _fill_missing_values(self, df: pd.DataFrame, fill_method: str = 'auto',
                             max_gap: int = 5, interpolate_prices: bool = True) -> pd.DataFrame:

        if df.empty:
            return df

        # Створимо копію для запобігання модифікації оригінального DataFrame
        df = df.copy()

        # Створимо маску пропущених значень для подальшого аналізу
        missing_mask = df.isna()

        # Підрахуємо кількість пропущених значень до заповнення для логування
        initial_missing = missing_mask.sum().sum()

        # Мапінг колонок до нижнього регістру для спрощення пошуку
        columns_lower = {col.lower(): col for col in df.columns}

        # Визначення колонок за категоріями
        price_cols = []
        volume_cols = []
        trades_cols = []
        other_numeric_cols = []
        non_numeric_cols = []

        # Визначаємо цінові колонки
        for col_pattern in ['open', 'high', 'low', 'close', 'vwap']:
            if col_pattern in columns_lower:
                price_cols.append(columns_lower[col_pattern])

        # Визначаємо колонки об'єму
        for col_pattern in ['volume', 'taker_buy_volume', 'taker_sell_volume', 'quote_volume',
                            'taker_buy_base_volume', 'taker_buy_quote_volume', 'quote_asset_volume']:
            if col_pattern in columns_lower:
                volume_cols.append(columns_lower[col_pattern])

        # Колонки для кількості угод
        for col_pattern in ['trades', 'number_of_trades']:
            if col_pattern in columns_lower:
                trades_cols.append(columns_lower[col_pattern])

        # Виділяємо всі інші числові колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        remaining_numeric = [col for col in numeric_cols if col not in price_cols and
                             col not in volume_cols and col not in trades_cols]

        # Виділяємо не-числові колонки
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

        # Функція для безпечного forward fill з обмеженням на max_gap
        def safe_ffill(series, limit=max_gap):
            return series.fillna(method='ffill', limit=limit)

        # Функція для безпечного backward fill з обмеженням на max_gap
        def safe_bfill(series, limit=max_gap):
            return series.fillna(method='bfill', limit=limit)

        # Заповнення цінових колонок
        if price_cols:
            if fill_method in ['auto', 'interpolate'] and interpolate_prices:
                # Спочатку спробуємо лінійну інтерполяцію для малих проміжків
                df[price_cols] = df[price_cols].interpolate(method='linear', limit=max_gap, limit_area='inside')

                # Для проміжків більших за max_gap або на краях використовуємо обмежений ffill/bfill
                df[price_cols] = df[price_cols].apply(safe_ffill)
                df[price_cols] = df[price_cols].apply(safe_bfill)
            elif fill_method in ['auto', 'ffill']:
                # Використовуємо безпечний ffill з обмеженням
                df[price_cols] = df[price_cols].apply(safe_ffill)
                df[price_cols] = df[price_cols].apply(safe_bfill)

        # Заповнення колонок об'єму
        if volume_cols:
            if fill_method in ['auto', 'zero']:
                # Для об'єму логічно використовувати нулі за відсутності даних
                df[volume_cols] = df[volume_cols].fillna(0)
            elif fill_method == 'ffill':
                df[volume_cols] = df[volume_cols].apply(safe_ffill)
                df[volume_cols] = df[volume_cols].fillna(0)  # Залишкові NaN як нулі

        # Заповнення колонок кількості угод
        if trades_cols:
            if fill_method in ['auto', 'zero']:
                df[trades_cols] = df[trades_cols].fillna(0)
            elif fill_method == 'ffill':
                df[trades_cols] = df[trades_cols].apply(safe_ffill)
                df[trades_cols] = df[trades_cols].fillna(0)

        # Заповнення інших числових колонок
        for col in remaining_numeric:
            col_lower = col.lower()
            if fill_method == 'auto':
                if any(x in col_lower for x in ['count', 'number', 'trades', 'qty', 'quantity', 'amount']):
                    # Лічильники заповнюємо нулями
                    df[col] = df[col].fillna(0)
                elif any(x in col_lower for x in ['funding', 'rate', 'fee']):
                    # Ставки заповнюємо обмеженим ffill або нулями
                    df[col] = safe_ffill(df[col]).fillna(0)
                else:
                    # Решта - обмежений ffill/bfill і нулі
                    df[col] = safe_ffill(df[col])
                    df[col] = safe_bfill(df[col])
                    df[col] = df[col].fillna(0)
            elif fill_method == 'ffill':
                df[col] = safe_ffill(df[col])
                df[col] = safe_bfill(df[col])
            elif fill_method == 'zero':
                df[col] = df[col].fillna(0)
            elif fill_method == 'interpolate':
                df[col] = df[col].interpolate(method='linear', limit=max_gap, limit_area='inside')
                df[col] = safe_ffill(df[col])
                df[col] = df[col].fillna(0)

        # Заповнення не-числових колонок (з обмеженням на max_gap)
        if non_numeric_cols and fill_method in ['auto', 'ffill', 'bfill']:
            for col in non_numeric_cols:
                df[col] = safe_ffill(df[col])
                df[col] = safe_bfill(df[col])

        # Підрахуємо кількість пропущених значень після заповнення для логування
        remaining_missing = df.isna().sum().sum()

        # Виявлення довгих проміжків пропусків, які можуть свідчити про проблеми з даними
        if initial_missing > 0:
            long_gaps = {}
            for col in df.columns:
                # Знаходимо проміжки NaN довші за max_gap
                gaps = missing_mask[col].astype(int).groupby(
                    (missing_mask[col] != missing_mask[col].shift()).cumsum()
                ).sum()
                long_gaps_count = (gaps[gaps > max_gap]).count()
                if long_gaps_count > 0:
                    long_gaps[col] = long_gaps_count

        return df


    def parse_interval(self, timeframe: str) -> pd.Timedelta:

        interval_map = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
        }

        import re
        match = re.match(r'(\d+)([smhdwM])', timeframe)
        if not match:
            raise ValueError(f"Неправильний формат інтервалу: {timeframe}")

        number, unit = match.groups()

        # Перевірка, чи число додатне
        if int(number) <= 0:
            raise ValueError(f"Інтервал повинен бути додатнім: {timeframe}")

        if unit == 'M':
            return pd.Timedelta(days=int(number) * 30.44)

        return pd.Timedelta(**{interval_map[unit]: int(number)})

    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:

        df = data.copy()

        # Зберігаємо оригінальні дані перед маніпуляціями
        self.original_data_map['data_before_time_features'] = df.copy()

        # Перевірка на DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame не має DatetimeIndex. Часові ознаки не можуть бути створені.")
            return df

        # Часові компоненти
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Додаємо is_weekend флаг (менш важливо для криптовалют, але залишаємо для повноти)
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday

        # Для криптовалют поняття торгової сесії менш важливе, але можна залишити
        # з точки зору активності різних регіонів
        df['session'] = 'Unknown'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'Asia'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'Europe'
        df.loc[(df['hour'] >= 16) & (df['hour'] < 24), 'session'] = 'US'

        # Додаємо ознаку "час доби" для криптовалют
        df['time_of_day'] = 'Unknown'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 6), 'time_of_day'] = 'Night'
        df.loc[(df['hour'] >= 6) & (df['hour'] < 12), 'time_of_day'] = 'Morning'
        df.loc[(df['hour'] >= 12) & (df['hour'] < 18), 'time_of_day'] = 'Afternoon'
        df.loc[(df['hour'] >= 18) & (df['hour'] < 24), 'time_of_day'] = 'Evening'

        # Циклічні ознаки для часових компонентів
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
        df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
        df['day_of_month_sin'] = np.sin((df['day_of_month'] - 1) * (2 * np.pi / 31))
        df['day_of_month_cos'] = np.cos((df['day_of_month'] - 1) * (2 * np.pi / 31))

        return df

    def make_stationary(self, data: pd.DataFrame, columns=None, method='diff',
                        order=1, seasonal_order=None) -> pd.DataFrame:
        self.logger.info(f"Наявні колонки в make_stationary: {list(data.columns)}")
        if data.empty:
            self.logger.warning("make_stationary: Отримано порожній DataFrame")
            return data

        if columns is None:
            columns = ['close']

        df = data.copy()
        result_df = df.copy()  # Створюємо окрему копію для результату, щоб не втратити оригінальні дані

        # Зберігаємо посилання на оригінальний DataFrame
        self.original_data_map['data_before_stationary'] = df.copy()

        # Створимо словник для відображення колонок за нижнім регістром
        column_map = {col.lower(): col for col in df.columns}

        # Для кожної колонки в переліку
        for col_name in columns:
            # Знаходимо відповідну колонку, незалежно від регістру
            col = None
            if col_name in df.columns:
                col = col_name
            elif col_name.lower() in column_map:
                col = column_map[col_name.lower()]

            if col is None:
                self.logger.warning(f"make_stationary: Колонка '{col_name}' відсутня у DataFrame і буде пропущена")
                continue

            # Зберігаємо оригінальні дані для цієї колонки
            col_key = f"{col}_original"
            self.original_data_map[col_key] = df[col].copy()
            self.logger.info(f"make_stationary: Збережено оригінальні дані для колонки '{col}'")

            # Перевірка на наявність нескінченних значень
            if np.isinf(df[col]).any():
                self.logger.warning(
                    f"make_stationary: Колонка {col} містить нескінченні значення, які будуть замінені на NaN")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # Перевірка на константні значення
            if df[col].nunique() <= 1:
                self.logger.warning(
                    f"make_stationary: Колонка {col} містить константні значення, що може призвести до проблем")

            # Базове диференціювання
            if method == 'diff' or method == 'all':
                diff_col = f'{col}_diff'
                result_df[diff_col] = df[col].diff(order)
                self.logger.info(f"make_stationary: Створено колонку {diff_col}")

                # Додаємо диференціювання 2-го порядку
                diff2_col = f'{col}_diff2'
                result_df[diff2_col] = df[col].diff().diff()
                self.logger.info(f"make_stationary: Створено колонку {diff2_col}")

                # Додаємо сезонне диференціювання якщо потрібно
                if seasonal_order and seasonal_order > 0:
                    seasonal_diff_col = f'{col}_seasonal_diff'
                    result_df[seasonal_diff_col] = df[col].diff(seasonal_order)
                    self.logger.info(f"make_stationary: Створено колонку {seasonal_diff_col}")

                    # Комбіноване сезонне + звичайне диференціювання
                    combo_diff_col = f'{col}_combo_diff'
                    result_df[combo_diff_col] = df[col].diff(seasonal_order).diff()
                    self.logger.info(f"make_stationary: Створено колонку {combo_diff_col}")

            # Логарифмічне перетворення - покращена обробка нульових та від'ємних значень
            if method == 'log' or method == 'all':
                # Створюємо копію даних для логарифмічної трансформації
                log_series = df[col].copy()

                # Перевірка на наявність нульових або від'ємних значень
                if (log_series <= 0).any():
                    self.logger.warning(
                        f"make_stationary: Колонка {col} містить нульові або від'ємні значення. "
                        f"Застосовуємо зсув перед логарифмічним перетворенням.")

                    # Знаходимо мінімальне значення для обчислення зсуву
                    min_val = log_series.min()

                    # Якщо мінімальне значення <= 0, додаємо зсув
                    if min_val <= 0:
                        # Зсув: мінімальне значення + 1 (щоб уникнути нуля)
                        shift = abs(min_val) + 1 if min_val < 0 else 1
                        log_series = log_series + shift
                        self.logger.info(
                            f"make_stationary: Застосовано зсув {shift} для логарифмічного перетворення колонки {col}")

                # Застосовуємо логарифм
                log_col = f'{col}_log'
                result_df[log_col] = np.log(log_series)
                self.logger.info(f"make_stationary: Створено колонку {log_col}")

                # Логарифм + диференціювання
                log_diff_col = f'{col}_log_diff'
                result_df[log_diff_col] = result_df[log_col].diff(order)
                self.logger.info(f"make_stationary: Створено колонку {log_diff_col}")

            # Відсоткова зміна - покращена обробка
            if method == 'pct_change' or method == 'all':
                pct_col = f'{col}_pct'
                # Перевірка на наявність послідовних нулів, які викликають NaN при pct_change
                zeros_count = (df[col] == 0).sum()
                if zeros_count > 0:
                    self.logger.warning(f"make_stationary: Колонка {col} містить {zeros_count} нульових значень, "
                                        f"що може призвести до NaN у відсотковій зміні")

                # Обчислюємо відсоткову зміну
                result_df[pct_col] = df[col].pct_change(order)
                self.logger.info(f"make_stationary: Створено колонку {pct_col}")

                # Для volume можемо спробувати альтернативний підхід якщо є багато нулів
                if col.lower() == 'volume' and zeros_count > len(df) * 0.1:  # якщо > 10% нулів
                    # Додаємо альтернативну версію з малим значенням замість нуля
                    pct_col_safe = f'{col}_pct_safe'
                    # Заміняємо нулі на малі значення перед обчисленням відсоткової зміни
                    safe_series = df[col].replace(0, 0.000001)
                    result_df[pct_col_safe] = safe_series.pct_change(order)
                    self.logger.info(f"make_stationary: Створено колонку {pct_col_safe} з безпечною заміною нулів")

            # Додаємо різницю між high та low (волатильність) якщо це колонка close
            high_col = self.find_column(df, 'high')
            low_col = self.find_column(df, 'low')

            if col.lower() == 'close' and high_col and low_col:
                high_low_range_col = 'high_low_range'
                result_df[high_low_range_col] = df[high_col] - df[low_col]
                self.logger.info(f"make_stationary: Створено колонку {high_low_range_col}")

                high_low_range_pct_col = 'high_low_range_pct'
                # Виправлення: перевірка на ділення на нуль
                mask = df[col] != 0  # Створюємо маску для ненульових значень
                result_df[high_low_range_pct_col] = np.nan  # Ініціалізуємо колонку як NaN

                # Застосовуємо ділення тільки там, де знаменник не дорівнює нулю
                if mask.any():  # Перевіряємо, що є ненульові значення
                    result_df.loc[mask, high_low_range_pct_col] = result_df.loc[mask, high_low_range_col] / df.loc[
                        mask, col]
                    self.logger.info(f"make_stationary: Створено колонку {high_low_range_pct_col}")

                    # Рахуємо скільки значень були замінені на NaN
                    na_count = (~mask).sum()
                    if na_count > 0:
                        self.logger.warning(
                            f"make_stationary: {na_count} значень, де {col} = 0, замінені на NaN в {high_low_range_pct_col}")
                else:
                    self.logger.warning(
                        f"make_stationary: Всі значення у колонці {col} дорівнюють 0, колонка {high_low_range_pct_col} буде містити лише NaN")

            # Для об'єму додаємо логарифм з покращеною обробкою нулів
            if col.lower() == 'volume':
                vol_series = df[col].copy()

                # Перевірка на нульові значення
                if (vol_series == 0).any():
                    zeros_count = (vol_series == 0).sum()
                    self.logger.warning(f"make_stationary: Колонка {col} містить {zeros_count} нульових значень")

                    # Додаємо малу константу до всіх значень для уникнення log(0)
                    vol_series = vol_series + 0.000001
                    self.logger.info(
                        f"make_stationary: Додано малу константу до колонки {col} для логарифмічного перетворення")

                vol_log_col = f'{col}_log'
                result_df[vol_log_col] = np.log(vol_series)
                self.logger.info(f"make_stationary: Створено колонку {vol_log_col}")

                vol_log_diff_col = f'{col}_log_diff'
                result_df[vol_log_diff_col] = result_df[vol_log_col].diff(order)
                self.logger.info(f"make_stationary: Створено колонку {vol_log_diff_col}")

        # Зберігаємо повний DataFrame до видалення NaN
        self.original_data_map['stationary_with_na'] = result_df.copy()

        # Перевіряємо наявність NaN
        na_count = result_df.isna().sum().sum()
        if na_count > 0:
            self.logger.info(f"make_stationary: У результуючому DataFrame виявлено {na_count} NaN значень")

            # Замість повного видалення NaN, можемо замінити їх на відповідні значення
            # Для збереження розмірності даних
            for col in result_df.columns:
                if result_df[col].isna().any():
                    na_count_col = result_df[col].isna().sum()
                    na_pct = na_count_col / len(result_df) * 100
                    self.logger.info(
                        f"make_stationary: Колонка {col} містить {na_count_col} NaN значень ({na_pct:.2f}%)")

                    if col.endswith(('_diff', '_diff2', '_seasonal_diff', '_combo_diff', '_log_diff', '_pct')):
                        # Для диференційованих даних заповнюємо NaN нулями
                        result_df[col] = result_df[col].fillna(0)
                        self.logger.info(f"make_stationary: NaN в {col} замінені на 0")
                    elif col.endswith('_log'):
                        # Для логарифмічних даних використовуємо ffill/bfill
                        # Спочатку перевіряємо скільки NaN послідовно
                        max_consecutive_na = result_df[col].isna().astype(int).groupby(
                            result_df[col].notna().astype(int).cumsum()).sum().max()

                        if max_consecutive_na > len(result_df) * 0.1:  # Якщо є довгі послідовності NaN
                            self.logger.warning(f"make_stationary: В колонці {col} виявлено довгі послідовності NaN "
                                                f"({max_consecutive_na} значень), що може вплинути на стаціонарність")

                        # Заповнюємо ffill потім bfill для крайніх випадків
                        result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
                        self.logger.info(f"make_stationary: NaN в {col} замінені методами ffill/bfill")

                    elif col == 'high_low_range_pct':  # Додаємо особливу обробку для high_low_range_pct
                        # Для high_low_range_pct заповнюємо NaN середнім значенням або нулем
                        if result_df[col].notna().any():  # Якщо є хоч якісь не-NaN значення
                            mean_val = result_df[col].mean()
                            result_df[col] = result_df[col].fillna(mean_val)
                            self.logger.info(f"make_stationary: NaN в {col} замінені на середнє значення {mean_val}")
                        else:
                            result_df[col] = result_df[col].fillna(0)
                            self.logger.info(f"make_stationary: NaN в {col} замінені на 0")
                    else:
                        # Для інших колонок, використовуємо інтерполяцію, якщо це можливо
                        try:
                            prev_na = result_df[col].isna().sum()
                            result_df[col] = result_df[col].interpolate(method='linear').fillna(method='ffill').fillna(
                                method='bfill')
                            after_na = result_df[col].isna().sum()
                            self.logger.info(
                                f"make_stationary: {prev_na - after_na} NaN в {col} замінені інтерполяцією")
                        except Exception as e:
                            # Якщо інтерполяція не вдалася, використовуємо просту заміну
                            result_df[col] = result_df[col].fillna(
                                result_df[col].median() if result_df[col].notna().any() else 0)
                            self.logger.warning(f"make_stationary: Помилка при інтерполяції {col}: {str(e)}, "
                                                f"використано медіану або 0")

            # Перевіряємо, чи всі NaN були замінені
            remaining_na = result_df.isna().sum().sum()
            if remaining_na > 0:
                self.logger.warning(f"make_stationary: Залишилось {remaining_na} NaN значень після заповнення")

                # Якщо залишились NaN, зберігаємо оригінальні дані й видаляємо рядки з NaN
                rows_before = len(result_df)
                cleaned_df = result_df.dropna()
                rows_after = len(cleaned_df)

                if rows_before > rows_after:
                    self.logger.warning(f"make_stationary: Видалено {rows_before - rows_after} рядків з NaN "
                                        f"після диференціювання")
                    # Зберігаємо індекси видалених рядків для можливого відновлення
                    dropped_indices = set(result_df.index) - set(cleaned_df.index)
                    self.original_data_map['dropped_indices'] = list(dropped_indices)

                result_df = cleaned_df

        # Перевірка на нескінченні значення після обробки
        if np.isinf(result_df.values).any():
            self.logger.warning("make_stationary: Виявлено нескінченні значення у результаті. Замінюємо на NaN.")
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            result_df = result_df.fillna(method='ffill').fillna(method='bfill')

        # Зберігаємо фінальний результат для подальшого використання
        self.original_data_map['stationary_result'] = result_df.copy()

        # Додаємо інформацію про стаціонарність для логування
        stationary_cols = []
        for col in result_df.columns:
            if col.endswith(('_diff', '_diff2', '_seasonal_diff', '_combo_diff', '_log_diff', '_pct')):
                stationary_cols.append(col)

        self.logger.info(f"make_stationary: Створено стаціонарні дані з {len(result_df)} рядками та "
                         f"{len(result_df.columns)} колонками. Стаціонарні колонки: {stationary_cols}")

        return result_df

    def check_stationarity(self, data: pd.DataFrame, column='close_diff', sample_size=10000,
                           parallel=True, confidence_level=0.05) -> dict:

        self.logger.info(f"Наявні колонки в check_stationarity: {list(data.columns)}")
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        import numpy as np
        import concurrent.futures

        results = {}

        # Перевірка наявності даних
        if data.empty:
            error_msg = "check_stationarity: Отримано порожній DataFrame для перевірки стаціонарності"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Перевіряємо наявність колонки, незалежно від регістру
        column_to_use = self.find_column(data, column)

        if column_to_use is None:
            error_msg = f"check_stationarity: Колонка '{column}' відсутня у DataFrame. Доступні колонки: {list(data.columns)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Розширена перевірка даних перед аналізом
        data_series = data[column_to_use].copy()

        # Перевірка на нескінченні значення
        inf_count = np.isinf(data_series).sum()
        if inf_count > 0:
            self.logger.warning(f"check_stationarity: Виявлено {inf_count} нескінченних значень. Замінюємо на NaN.")
            data_series = data_series.replace([np.inf, -np.inf], np.nan)

        # Перевірка на відсутні значення
        na_count = data_series.isna().sum()
        if na_count > 0:
            self.logger.warning(f"check_stationarity: Виявлено {na_count} NaN значень. Видаляємо їx.")

        # Очищення даних від NaN
        clean_data = data_series.dropna()

        if len(clean_data) < 10:  # Потрібно більше точок для надійних тестів
            error_msg = f"check_stationarity: Недостатньо даних для перевірки стаціонарності після видалення NaN значень. Залишилося точок: {len(clean_data)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Перевірка на константне значення або близьке до константного
        unique_values = clean_data.nunique()
        if unique_values <= 1:
            error_msg = f"check_stationarity: Дані в колонці '{column_to_use}' є константою. Такі дані не можуть бути стаціонарними."
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Перевірка на дуже малу варіацію
        std_dev = clean_data.std()
        mean_val = clean_data.mean()
        if std_dev == 0 or (std_dev / abs(mean_val) < 1e-6 and mean_val != 0):
            error_msg = f"check_stationarity: Дані в колонці '{column_to_use}' мають дуже малу варіацію (std={std_dev}, mean={mean_val})."
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Використання вибірки для великих наборів даних
        data_size = len(clean_data)
        self.logger.info(f"check_stationarity: Розмір даних - {data_size} точок")

        # Якщо даних більше ніж sample_size, беремо вибірку
        if data_size > sample_size:
            # Стратегія вибірки: беремо рівномірно розподілені точки з усього ряду
            step = max(1, data_size // sample_size)
            sampled_data = clean_data.iloc[::step].copy()
            self.logger.info(f"check_stationarity: Використовуємо вибірку {len(sampled_data)} точок (крок {step})")
        else:
            sampled_data = clean_data.copy()

        # Додаткова перевірка даних у вибірці
        if sampled_data.var() == 0:
            error_msg = f"check_stationarity: Дані у вибірці не мають варіації (константа)."
            self.logger.error(error_msg)
            return {'error': error_msg, 'is_stationary': False}

        # Функції для виконання тестів
        def run_adf_test():
            try:
                # Використовуємо менше лагів для прискорення
                # Формула для автоматичного визначення максимальної кількості лагів
                max_lags = min(int(np.ceil(12 * (len(sampled_data) / 100) ** (1 / 4))), 20)

                # Додаємо обробку помилок у випадку збою тесту
                try:
                    adf_result = adfuller(sampled_data, maxlag=max_lags)
                    return {
                        'test_statistic': adf_result[0],
                        'p-value': adf_result[1],
                        'is_stationary': adf_result[1] < confidence_level,
                        'critical_values': adf_result[4],
                        'used_lags': max_lags
                    }
                except Exception as e:
                    # Якщо перша спроба не вдалася, спробуємо використати менше лагів
                    self.logger.warning(
                        f"ADF тест з {max_lags} лагами не вдався: {str(e)}. Спробуємо з меншою кількістю лагів.")

                    # Зменшуємо кількість лагів і пробуємо знову
                    reduced_lags = max(1, max_lags // 2)
                    adf_result = adfuller(sampled_data, maxlag=reduced_lags)
                    return {
                        'test_statistic': adf_result[0],
                        'p-value': adf_result[1],
                        'is_stationary': adf_result[1] < confidence_level,
                        'critical_values': adf_result[4],
                        'used_lags': reduced_lags,
                        'warning': f"Використано зменшену кількість лагів: {reduced_lags}"
                    }

            except Exception as e:
                return {
                    'error': f"Помилка при виконанні ADF тесту: {str(e)}",
                    'is_stationary': False
                }

        def run_kpss_test():
            try:
                # Використовуємо менше лагів для прискорення
                # Формула для автоматичного визначення максимальної кількості лагів
                max_lags = min(int(np.ceil(12 * (len(sampled_data) / 100) ** (1 / 4))), 20)

                try:
                    kpss_result = kpss(sampled_data, nlags=max_lags)
                    return {
                        'test_statistic': kpss_result[0],
                        'p-value': kpss_result[1],
                        'is_stationary': kpss_result[1] > confidence_level,
                        'critical_values': kpss_result[3],
                        'used_lags': max_lags
                    }
                except Exception as e:
                    # Якщо перша спроба не вдалася, спробуємо використати менше лагів
                    self.logger.warning(
                        f"KPSS тест з {max_lags} лагами не вдався: {str(e)}. Спробуємо з меншою кількістю лагів.")

                    # Зменшуємо кількість лагів і пробуємо знову
                    reduced_lags = max(1, max_lags // 2)
                    kpss_result = kpss(sampled_data, nlags=reduced_lags)
                    return {
                        'test_statistic': kpss_result[0],
                        'p-value': kpss_result[1],
                        'is_stationary': kpss_result[1] > confidence_level,
                        'critical_values': kpss_result[3],
                        'used_lags': reduced_lags,
                        'warning': f"Використано зменшену кількість лагів: {reduced_lags}"
                    }

            except Exception as e:
                return {
                    'error': f"Помилка при виконанні KPSS тесту: {str(e)}",
                    'is_stationary': True  # За замовчуванням вважаємо стаціонарним, якщо KPSS не може бути виконаний
                }

        def run_acf_pacf_analysis():
            try:
                # Визначаємо розумну кількість лагів
                # Для 4 млн точок, навіть sample_size може бути надто великим для ACF/PACF
                acf_pacf_sample = sampled_data
                if len(sampled_data) > 2000:
                    # Для ACF/PACF використовуємо ще меншу вибірку
                    step = max(1, len(sampled_data) // 2000)
                    acf_pacf_sample = sampled_data.iloc[::step].copy()
                    self.logger.info(
                        f"ACF/PACF: Використовуємо зменшену вибірку {len(acf_pacf_sample)} точок (крок {step})")

                # Визначаємо кількість лагів для аналізу (не більше 50 або 25% від розміру вибірки)
                nlags = min(50, len(acf_pacf_sample) // 4)
                nlags = max(10, nlags)  # Але не менше 10

                # Обчислюємо ACF і PACF
                acf_values = acf(acf_pacf_sample, nlags=nlags, fft=True)
                pacf_values = pacf(acf_pacf_sample, nlags=nlags, method='ols')

                # Аналіз отриманих значень ACF/PACF
                acf_decays = np.all(np.abs(acf_values[1:]) < np.abs(acf_values[:-1]))
                significant_pacf = np.sum(np.abs(pacf_values[1:]) > 1.96 / np.sqrt(len(acf_pacf_sample)))

                # Оцінка стаціонарності на основі ACF/PACF
                is_stationary_acf = acf_decays and (significant_pacf < nlags // 3)

                return {
                    'acf_decays': acf_decays,
                    'significant_pacf_count': significant_pacf,
                    'is_stationary': is_stationary_acf,
                    'nlags_used': nlags,
                    'sample_size': len(acf_pacf_sample)
                }
            except Exception as e:
                return {
                    'error': f"Помилка при виконанні ACF/PACF аналізу: {str(e)}",
                    'is_stationary': None  # Недостатньо інформації для визначення стаціонарності
                }

        # Виконання тестів
        if parallel:
            try:
                # Паралельне виконання тестів для прискорення
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {
                        'adf': executor.submit(run_adf_test),
                        'kpss': executor.submit(run_kpss_test),
                        'acf_pacf': executor.submit(run_acf_pacf_analysis)
                    }

                    # Збір результатів
                    for name, future in futures.items():
                        try:
                            results[name] = future.result()
                        except Exception as e:
                            results[name] = {
                                'error': f"Помилка при виконанні {name} тесту: {str(e)}",
                                'is_stationary': None
                            }
            except Exception as e:
                self.logger.error(f"Помилка при паралельному виконанні тестів: {str(e)}")
                # Якщо паралельне виконання не вдалося, спробуємо послідовно
                parallel = False

        # Послідовне виконання тестів, якщо паралельне не вдалося або не запитано
        if not parallel:
            self.logger.info("check_stationarity: Виконуємо тести послідовно")
            results['adf'] = run_adf_test()
            results['kpss'] = run_kpss_test()
            results['acf_pacf'] = run_acf_pacf_analysis()

        # Аналіз узгодженості результатів тестів
        adf_stationary = results.get('adf', {}).get('is_stationary', False)
        kpss_stationary = results.get('kpss', {}).get('is_stationary', True)
        acf_pacf_stationary = results.get('acf_pacf', {}).get('is_stationary', None)

        self.logger.info(f"check_stationarity: Результати тестів: "
                         f"ADF: {'Стаціонарний' if adf_stationary else 'Нестаціонарний'}, "
                         f"KPSS: {'Стаціонарний' if kpss_stationary else 'Нестаціонарний'}, "
                         f"ACF/PACF: {'Стаціонарний' if acf_pacf_stationary == True else 'Нестаціонарний' if acf_pacf_stationary == False else 'Невизначено'}")

        # Визначення загального висновку про стаціонарність
        # Класифікуємо ряд як стаціонарний, якщо обидва основні тести (ADF і KPSS) узгоджуються
        # або якщо ADF показує стаціонарність і ACF/PACF підтверджує
        if adf_stationary and kpss_stationary:
            is_stationary = True
            confidence = "висока"
        elif not adf_stationary and not kpss_stationary:
            is_stationary = False
            confidence = "висока"
        elif adf_stationary and acf_pacf_stationary:
            is_stationary = True
            confidence = "середня"
        elif not adf_stationary and acf_pacf_stationary is False:
            is_stationary = False
            confidence = "середня"
        else:
            # Якщо тести не узгоджуються, віддаємо перевагу ADF, але з низькою впевненістю
            is_stationary = adf_stationary
            confidence = "низька"

        # Формування підсумкового результату
        final_result = {
            'is_stationary': is_stationary,
            'confidence': confidence,
            'column': column_to_use,
            'tests': results,
            'data_size': data_size,
            'sample_size': len(sampled_data)
        }

        # Додаємо статистичні характеристики даних
        final_result['stats'] = {
            'mean': float(clean_data.mean()),
            'std': float(clean_data.std()),
            'min': float(clean_data.min()),
            'max': float(clean_data.max()),
            'unique_values': int(unique_values)
        }

        # Логування висновку
        self.logger.info(
            f"check_stationarity: Висновок - ряд '{column_to_use}' {'є стаціонарним' if is_stationary else 'не є стаціонарним'} "
            f"з {confidence} впевненістю.")

        return final_result

    def prepare_arima_data(
            self,
            data: pd.DataFrame | dd.DataFrame,
            symbol: str,
            timeframe: str,
            sample_size: int = 10000,
            parallel: bool = True
    ) -> pd.DataFrame:

        try:
            self.logger.info(f"prepare_arima_data: Початок підготовки даних для {symbol} ({timeframe})")

            # Конвертація Dask DataFrame у pandas при необхідності
            if hasattr(data, 'compute'):
                self.logger.info("prepare_arima_data: Конвертація Dask DataFrame у pandas DataFrame")
                data = data.compute()

            if data.empty:
                self.logger.error("prepare_arima_data: Отримано порожній DataFrame")
                return pd.DataFrame()

            # Забезпечення наявності DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'open_time' in data.columns:
                    self.logger.info("prepare_arima_data: Встановлення 'open_time' як індекс")
                    data.set_index('open_time', inplace=True)
                else:
                    self.logger.error("prepare_arima_data: Не знайдено індекс datetime або колонку open_time")
                    return pd.DataFrame()

            # 1. Знаходження колонки з цінами закриття
            close_columns = ['close', 'price', 'last', 'last_price']
            close_column = next((col for col in close_columns if col in data.columns), None)

            if not close_column:
                close_column = next((col for col in data.columns if 'close' in col.lower()), None)

            if not close_column:
                self.logger.error("prepare_arima_data: Не знайдено колонку з цінами закриття")
                return pd.DataFrame()

            self.logger.info(f"prepare_arima_data: Використовуємо колонку '{close_column}' як ціну закриття")

            # Створення копії для результатів
            result_df = pd.DataFrame(index=data.index)

            # Базова інформація
            result_df['timeframe'] = timeframe
            result_df['original_close'] = data[close_column]

            # Перевірка на нескінченні значення
            if np.isinf(result_df['original_close']).any():
                inf_count = np.isinf(result_df['original_close']).sum()
                self.logger.warning(f"prepare_arima_data: Виявлено {inf_count} нескінченних значень. Замінюємо на NaN.")
                result_df['original_close'] = result_df['original_close'].replace([np.inf, -np.inf], np.nan)

            # Перевірка на відсутні значення
            na_count = result_df['original_close'].isna().sum()
            if na_count > 0:
                self.logger.warning(f"prepare_arima_data: Виявлено {na_count} NaN значень.")
                # Застосовуємо інтерполяцію для заповнення пропусків
                result_df['original_close'] = result_df['original_close'].interpolate(method='linear').fillna(
                    method='ffill').fillna(method='bfill')

                remaining_na = result_df['original_close'].isna().sum()
                if remaining_na > 0:
                    self.logger.warning(
                        f"prepare_arima_data: Після інтерполяції залишилось {remaining_na} NaN значень.")

            # 2. Розширений набір трансформацій для забезпечення стаціонарності

            # Базові диференціювання
            result_df['close_diff'] = result_df['original_close'].diff()
            result_df['close_diff2'] = result_df['close_diff'].diff()

            # Логарифмічне перетворення з обробкою нульових та від'ємних значень
            close_series = result_df['original_close'].copy()

            # Перевірка на наявність нульових або від'ємних значень
            if (close_series <= 0).any():
                zeros_count = (close_series <= 0).sum()
                self.logger.warning(
                    f"prepare_arima_data: Виявлено {zeros_count} нульових або від'ємних значень. "
                    f"Застосовуємо зсув перед логарифмічним перетворенням.")

                # Зсув для безпечного логарифмування
                min_val = close_series.min()
                shift = abs(min_val) + 1 if min_val <= 0 else 1
                close_series = close_series + shift
                self.logger.info(f"prepare_arima_data: Застосовано зсув {shift} для логарифмічного перетворення")

            result_df['close_log'] = np.log(close_series)
            result_df['close_log_diff'] = result_df['close_log'].diff()

            # Відсоткова зміна
            result_df['close_pct_change'] = result_df['original_close'].pct_change()

            # Визначення сезонного періоду і сезонне диференціювання
            season_period = self._determine_seasonal_period(timeframe)
            result_df['close_seasonal_diff'] = result_df['original_close'].diff(season_period)
            result_df['close_combo_diff'] = result_df['close_seasonal_diff'].diff()

            # Додаємо обробку для високо-низько діапазону, якщо дані доступні
            high_col = self.find_column(data, 'high')
            low_col = self.find_column(data, 'low')

            if high_col and low_col:
                self.logger.info(
                    f"prepare_arima_data: Знайдено колонки '{high_col}' та '{low_col}', обчислюємо діапазон")
                result_df['high_low_range'] = data[high_col] - data[low_col]

                # Волатильність як відсоток від ціни
                mask = result_df['original_close'] != 0
                result_df['high_low_range_pct'] = np.nan

                # Застосовуємо ділення тільки там, де знаменник не дорівнює нулю
                if mask.any():
                    result_df.loc[mask, 'high_low_range_pct'] = result_df.loc[mask, 'high_low_range'] / result_df.loc[
                        mask, 'original_close']

                na_count = (~mask).sum()
                if na_count > 0:
                    self.logger.warning(
                        f"prepare_arima_data: {na_count} значень у high_low_range_pct замінені на NaN через ділення на нуль")

            # У секції обробки volume
            volume_col = self.find_column(data, 'volume')
            self.logger.info(f"prepare_arima_data: Результат пошуку колонки volume: {volume_col}")
            if volume_col:
                self.logger.info(
                    f"prepare_arima_data: Знайдено колонку '{volume_col}', додаємо розширені трансформації об'єму")

                # Явне перетворення до числового типу з діагностичним виводом
                original_vol_series = data[volume_col].copy()
                self.logger.info(
                    f"prepare_arima_data: Тип даних оригінальної серії volume: {original_vol_series.dtype}")
                self.logger.info(
                    f"prepare_arima_data: Перші 5 значень оригінальної серії volume: {original_vol_series.head(5).tolist()}")
                self.logger.info(
                    f"prepare_arima_data: Статистика оригінальної серії volume: min={original_vol_series.min()}, max={original_vol_series.max()}")

                # Безпечне перетворення до числового типу
                vol_series = pd.to_numeric(original_vol_series, errors='coerce')
                self.logger.info(f"prepare_arima_data: Тип даних після перетворення: {vol_series.dtype}")
                self.logger.info(f"prepare_arima_data: Кількість NaN після перетворення: {vol_series.isna().sum()}")

                # Перевірка, чи серія має дійсні дані
                if vol_series.isna().all():
                    self.logger.error(f"prepare_arima_data: Всі значення volume стали NaN після перетворення!")
                    # Спробуємо альтернативний підхід - використовувати оригінальні дані
                    vol_series = original_vol_series

                # Зберігаємо оригінальний volume (для діагностики)
                result_df['original_volume_raw'] = original_vol_series
                result_df['original_volume'] = vol_series

                # Явно перевіряємо і логуємо вставлені значення
                self.logger.info(
                    f"prepare_arima_data: Перевірка result_df['original_volume']: Тип - {result_df['original_volume'].dtype}")
                self.logger.info(
                    f"prepare_arima_data: Кількість NaN у result_df['original_volume']: {result_df['original_volume'].isna().sum()}")
                self.logger.info(
                    f"prepare_arima_data: Перші 5 значень result_df['original_volume']: {result_df['original_volume'].head(5).tolist()}")

                # Безпечна перевірка на нульові значення
                has_zeros = (vol_series == 0).any()
                self.logger.info(f"prepare_arima_data: Серія має нулі: {has_zeros}")

                # Безпечно створюємо копію для подальшої обробки
                vol_series_safe = vol_series.copy()

                # Безпечно замінюємо нулі
                if has_zeros:
                    zeros_count = (vol_series_safe == 0).sum()
                    self.logger.warning(
                        f"prepare_arima_data: Колонка '{volume_col}' містить {zeros_count} нульових значень")

                    # Додаємо малу константу для уникнення log(0)
                    vol_series_safe = vol_series_safe.replace(0, 0.000001)

                    # Перевірка після заміни
                    self.logger.info(
                        f"prepare_arima_data: Кількість нулів після заміни: {(vol_series_safe == 0).sum()}")

                # Основні трансформації об'єму з додатковою діагностикою
                try:
                    result_df['volume_log'] = np.log(vol_series_safe)
                    self.logger.info(
                        f"prepare_arima_data: Створено volume_log, кількість NaN: {result_df['volume_log'].isna().sum()}")

                    result_df['volume_diff'] = vol_series_safe.diff()
                    self.logger.info(
                        f"prepare_arima_data: Створено volume_diff, кількість NaN: {result_df['volume_diff'].isna().sum()}")

                    result_df['volume_log_diff'] = result_df['volume_log'].diff()
                    self.logger.info(
                        f"prepare_arima_data: Створено volume_log_diff, кількість NaN: {result_df['volume_log_diff'].isna().sum()}")

                    result_df['volume_pct_change'] = vol_series_safe.pct_change()
                    self.logger.info(
                        f"prepare_arima_data: Створено volume_pct_change, кількість NaN: {result_df['volume_pct_change'].isna().sum()}")

                    # Додаємо сезонну різницю для об'єму
                    result_df['volume_seasonal_diff'] = vol_series_safe.diff(season_period)
                    self.logger.info(
                        f"prepare_arima_data: Створено volume_seasonal_diff, кількість NaN: {result_df['volume_seasonal_diff'].isna().sum()}")

                    # Додаткові трансформації для об'єму
                    # Безпечно обчислюємо window_size
                    window_size = min(20, len(vol_series_safe) // 10) if len(vol_series_safe) > 20 else 5
                    self.logger.info(
                        f"prepare_arima_data: Використовуємо window_size={window_size} для обчислення volume_relative")

                    # Безпечно обчислюємо середнє значення
                    rolling_mean = vol_series_safe.rolling(window=window_size).mean()
                    self.logger.info(
                        f"prepare_arima_data: Обчислено rolling_mean, кількість NaN: {rolling_mean.isna().sum()}")

                    # Обробляємо випадок ділення на нуль при обчисленні відносного значення
                    result_df['volume_relative'] = pd.Series(np.nan,
                                                             index=vol_series_safe.index)  # Ініціалізуємо пустими значеннями
                    non_zero_mask = rolling_mean != 0
                    self.logger.info(f"prepare_arima_data: Кількість нулів у rolling_mean: {(~non_zero_mask).sum()}")

                    if non_zero_mask.any():
                        result_df.loc[non_zero_mask, 'volume_relative'] = vol_series_safe.loc[non_zero_mask] / \
                                                                          rolling_mean.loc[non_zero_mask]

                    self.logger.info(
                        f"prepare_arima_data: Створено volume_relative, кількість NaN: {result_df['volume_relative'].isna().sum()}")

                    # Безпечно обчислюємо логарифм
                    result_df['volume_relative_log'] = np.log(
                        result_df['volume_relative'].replace([0, np.nan], 0.000001))
                    self.logger.info(
                        f"prepare_arima_data: Створено volume_relative_log, кількість NaN: {result_df['volume_relative_log'].isna().sum()}")

                except Exception as ex:
                    self.logger.error(f"prepare_arima_data: Помилка при створенні трансформацій volume: {str(ex)}")
                    import traceback
                    self.logger.error(f"prepare_arima_data: Деталі помилки: {traceback.format_exc()}")
            # Заповнення NA у всіх нових колонках
            for col in result_df.columns:
                if col not in ['timeframe']:
                    na_count = result_df[col].isna().sum()
                    if na_count > 0:
                        if col.endswith(('_diff', '_diff2', '_seasonal_diff', '_combo_diff', '_log_diff', '_pct',
                                         '_pct_change', '_relative', '_relative_log')):
                            # Для диференційованих даних заповнюємо NaN нулями
                            result_df[col] = result_df[col].fillna(0)
                        else:
                            # Для інших використовуємо інтерполяцію/ffill/bfill
                            result_df[col] = result_df[col].interpolate(method='linear').fillna(method='ffill').fillna(
                                method='bfill')

            # 3. Виконання тестів стаціонарності з покращеною реалізацією
            from statsmodels.tsa.stattools import adfuller, kpss
            import concurrent.futures
            import json

            # Визначаємо набір колонок для тестування стаціонарності
            stationary_columns = [
                'close_diff', 'close_diff2', 'close_log_diff', 'close_pct_change',
                'close_seasonal_diff', 'close_combo_diff'
            ]

            # Додаємо колонки з об'ємом, якщо вони існують
            volume_cols = [col for col in result_df.columns if col.startswith('volume_') and col != 'original_volume']
            if volume_cols:
                stationary_columns.extend(volume_cols)

            # Функції для тестів стаціонарності
            def run_adf_test(data_series):
                try:
                    # Вибірка і очищення даних
                    clean_series = data_series.dropna()
                    if len(clean_series) < 10:
                        return {'error': 'Недостатньо даних', 'is_stationary': None}

                    # Використання вибірки для великих наборів даних
                    if len(clean_series) > sample_size:
                        step = max(1, len(clean_series) // sample_size)
                        sampled_series = clean_series.iloc[::step]
                    else:
                        sampled_series = clean_series

                    # Формула для автоматичного визначення кількості лагів
                    max_lags = min(int(np.ceil(12 * (len(sampled_series) / 100) ** (1 / 4))), 20)

                    # Виконання тесту
                    try:
                        adf_result = adfuller(sampled_series, maxlag=max_lags)
                        return {
                            'test_statistic': adf_result[0],
                            'p-value': adf_result[1],
                            'is_stationary': adf_result[1] < 0.05,
                            'critical_values': adf_result[4],
                        }
                    except Exception as e:
                        # Спроба з меншою кількістю лагів
                        reduced_lags = max(1, max_lags // 2)
                        adf_result = adfuller(sampled_series, maxlag=reduced_lags)
                        return {
                            'test_statistic': adf_result[0],
                            'p-value': adf_result[1],
                            'is_stationary': adf_result[1] < 0.05,
                            'critical_values': adf_result[4],
                            'warning': f"Використано зменшену кількість лагів: {reduced_lags}"
                        }
                except Exception as e:
                    return {'error': f"Помилка ADF тесту: {str(e)}", 'is_stationary': None}

            def run_kpss_test(data_series):
                try:
                    # Вибірка і очищення даних
                    clean_series = data_series.dropna()
                    if len(clean_series) < 10:
                        return {'error': 'Недостатньо даних', 'is_stationary': None}

                    # Використання вибірки для великих наборів даних
                    if len(clean_series) > sample_size:
                        step = max(1, len(clean_series) // sample_size)
                        sampled_series = clean_series.iloc[::step]
                    else:
                        sampled_series = clean_series

                    # Формула для автоматичного визначення кількості лагів
                    max_lags = min(int(np.ceil(12 * (len(sampled_series) / 100) ** (1 / 4))), 20)

                    # Виконання тесту
                    try:
                        kpss_result = kpss(sampled_series, nlags=max_lags)
                        return {
                            'test_statistic': kpss_result[0],
                            'p-value': kpss_result[1],
                            'is_stationary': kpss_result[1] > 0.05,
                            'critical_values': kpss_result[3],
                        }
                    except Exception as e:
                        # Спроба з меншою кількістю лагів
                        reduced_lags = max(1, max_lags // 2)
                        kpss_result = kpss(sampled_series, nlags=reduced_lags)
                        return {
                            'test_statistic': kpss_result[0],
                            'p-value': kpss_result[1],
                            'is_stationary': kpss_result[1] > 0.05,
                            'critical_values': kpss_result[3],
                            'warning': f"Використано зменшену кількість лагів: {reduced_lags}"
                        }
                except Exception as e:
                    return {'error': f"Помилка KPSS тесту: {str(e)}", 'is_stationary': True}

            # Словник для зберігання результатів тестів
            stationarity_results = {}

            # Виконання тестів для кожної колонки (паралельно або послідовно)
            if parallel:
                try:
                    self.logger.info("prepare_arima_data: Виконуємо тести стаціонарності паралельно")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(stationary_columns))) as executor:
                        futures = {}

                        # Створюємо завдання для виконання тестів
                        for col in stationary_columns:
                            if col in result_df.columns:
                                futures[f"{col}_adf"] = executor.submit(run_adf_test, result_df[col])
                                futures[f"{col}_kpss"] = executor.submit(run_kpss_test, result_df[col])

                        # Збираємо результати
                        for future_key, future in futures.items():
                            col, test_type = future_key.rsplit('_', 1)
                            if col not in stationarity_results:
                                stationarity_results[col] = {}
                            stationarity_results[col][test_type] = future.result()
                except Exception as e:
                    self.logger.error(f"Помилка при паралельному виконанні тестів: {str(e)}")
                    parallel = False

            # Послідовне виконання, якщо паралельне не вдалося
            if not parallel:
                self.logger.info("prepare_arima_data: Виконуємо тести стаціонарності послідовно")
                for col in stationary_columns:
                    if col in result_df.columns:
                        stationarity_results[col] = {
                            'adf': run_adf_test(result_df[col]),
                            'kpss': run_kpss_test(result_df[col])
                        }

            # Аналіз результатів і визначення найкращої трансформації
            best_transformation = None
            best_confidence = "низька"
            best_p_value = 1.0

            # Зберігаємо загальні результати тестів для БД
            result_df['adf_pvalue'] = None
            result_df['kpss_pvalue'] = None
            result_df['is_stationary'] = False

            for col, tests in stationarity_results.items():
                adf_stationary = tests.get('adf', {}).get('is_stationary', False)
                adf_p_value = tests.get('adf', {}).get('p-value', 1.0)
                kpss_stationary = tests.get('kpss', {}).get('is_stationary', True)
                kpss_p_value = tests.get('kpss', {}).get('p-value', 1.0)

                # Визначення рівня впевненості в стаціонарності
                if adf_stationary and kpss_stationary:
                    confidence = "висока"
                    is_stationary = True
                elif not adf_stationary and not kpss_stationary:
                    confidence = "низька"  # Обидва тести показують нестаціонарність
                    is_stationary = False
                else:
                    confidence = "середня"
                    is_stationary = adf_stationary  # Пріоритет ADF тесту

                # Зберігаємо результат у DataFrame
                result_df[f"{col}_adf_stationary"] = adf_stationary
                result_df[f"{col}_adf_pvalue"] = adf_p_value
                result_df[f"{col}_kpss_stationary"] = kpss_stationary
                result_df[f"{col}_kpss_pvalue"] = kpss_p_value
                result_df[f"{col}_confidence"] = confidence
                result_df[f"{col}_is_stationary"] = is_stationary

                self.logger.info(f"prepare_arima_data: Колонка {col}: "
                                 f"ADF {'стаціонарна' if adf_stationary else 'нестаціонарна'} (p={adf_p_value:.5f}), "
                                 f"KPSS {'стаціонарна' if kpss_stationary else 'нестаціонарна'} (p={kpss_p_value:.5f}), "
                                 f"Впевненість: {confidence}")

                # Визначення найкращої трансформації (пріоритет: висока впевненість і низьке p-значення ADF)
                if adf_stationary:
                    confidence_priority = {"висока": 3, "середня": 2, "низька": 1}
                    current_priority = confidence_priority.get(confidence, 0)
                    best_priority = confidence_priority.get(best_confidence, 0)

                    if (current_priority > best_priority) or (
                            current_priority == best_priority and adf_p_value < best_p_value):
                        best_transformation = col
                        best_confidence = confidence
                        best_p_value = adf_p_value

            # Додаємо інформацію про найкращу трансформацію
            if best_transformation:
                result_df['best_transformation'] = best_transformation
                result_df['best_confidence'] = best_confidence
                result_df['best_adf_pvalue'] = best_p_value

                # Зберігаємо значення для БД з найкращої трансформації
                result_df['adf_pvalue'] = result_df[f"{best_transformation}_adf_pvalue"]
                result_df['kpss_pvalue'] = result_df[f"{best_transformation}_kpss_pvalue"]
                result_df['is_stationary'] = result_df[f"{best_transformation}_is_stationary"]

                self.logger.info(f"prepare_arima_data: Найкраща трансформація: {best_transformation} "
                                 f"(впевненість: {best_confidence}, adf_p={best_p_value:.5f})")
            else:
                result_df['best_transformation'] = 'close_diff'  # За замовчуванням
                result_df['best_confidence'] = 'невизначена'
                result_df['best_adf_pvalue'] = None
                self.logger.warning(
                    "prepare_arima_data: Не знайдено стаціонарних трансформацій, використовуємо close_diff за замовчуванням")

            # 4. Обчислення ACF/PACF для визначення параметрів моделі
            try:
                from statsmodels.tsa.stattools import acf, pacf

                # Використовуємо найкращу трансформацію або close_diff за замовчуванням
                series_for_acf = result_df.get(best_transformation, result_df['close_diff']).dropna()

                # Вибірка для великих наборів даних
                if len(series_for_acf) > 2000:
                    step = max(1, len(series_for_acf) // 2000)
                    series_for_acf = series_for_acf.iloc[::step]
                    self.logger.info(f"prepare_arima_data: ACF/PACF використовує вибірку з {len(series_for_acf)} точок")

                # Визначення кількості лагів
                nlags = min(50, len(series_for_acf) // 4)
                nlags = max(10, nlags)  # Але не менше 10

                # Обчислення ACF і PACF
                acf_values = acf(series_for_acf, nlags=nlags, fft=True)
                pacf_values = pacf(series_for_acf, nlags=nlags, method='ols')

                # Знаходження значущих лагів
                significant_lags_acf = []
                significant_lags_pacf = []
                confidence_level = 1.96 / np.sqrt(len(series_for_acf))

                for i in range(1, len(acf_values)):
                    if abs(acf_values[i]) > confidence_level:
                        significant_lags_acf.append(i)
                    if abs(pacf_values[i]) > confidence_level:
                        significant_lags_pacf.append(i)

                # Зберігаємо всі значущі лаги в JSON форматі для БД
                significant_lags = {
                    'acf': significant_lags_acf,
                    'pacf': significant_lags_pacf
                }
                result_df['significant_lags'] = json.dumps(significant_lags)

                # Визначення параметрів ARIMA
                # p: кількість значущих PACF лагів
                # d: порядок диференціювання
                # q: кількість значущих ACF лагів

                # Визначення d на основі трансформації
                if best_transformation in ['close_diff', 'close_log_diff', 'close_pct_change', 'volume_diff',
                                           'volume_log_diff', 'volume_pct_change']:
                    d = 1
                elif best_transformation in ['close_diff2']:
                    d = 2
                else:
                    d = 1  # За замовчуванням

                # Визначення p і q
                p = min(len(significant_lags_pacf), 5) if significant_lags_pacf else 0
                q = min(len(significant_lags_acf), 5) if significant_lags_acf else 0

                # Забезпечення мінімальних значень
                p = max(p, 1)
                q = max(q, 1)

                # Зберігаємо результати
                result_df['suggested_p'] = p
                result_df['suggested_d'] = d
                result_df['suggested_q'] = q
                result_df['significant_lags_acf'] = str(significant_lags_acf)
                result_df['significant_lags_pacf'] = str(significant_lags_pacf)

                self.logger.info(f"prepare_arima_data: Рекомендовані параметри ARIMA: ({p},{d},{q})")

                # Визначення сезонних параметрів
                if season_period > 1:
                    # Спрощений підхід для сезонних параметрів
                    result_df['suggested_seasonal_p'] = 1
                    result_df['suggested_seasonal_d'] = 1
                    result_df['suggested_seasonal_q'] = 1
                    result_df['suggested_seasonal_period'] = season_period

                    self.logger.info(
                        f"prepare_arima_data: Рекомендовані сезонні параметри: ({1},{1},{1})_{season_period}")
            except Exception as e:
                self.logger.error(f"Помилка при обчисленні ACF/PACF: {str(e)}")
                result_df['suggested_p'] = 1
                result_df['suggested_d'] = 1
                result_df['suggested_q'] = 1
                result_df['significant_lags'] = '{"acf":[], "pacf":[]}'

            # 5. Додаткові метрики, які необхідні для БД
            result_df['residual_variance'] = None  # Буде заповнено після підгонки моделі
            result_df['aic_score'] = None  # Буде заповнено після підгонки моделі
            result_df['bic_score'] = None  # Буде заповнено після підгонки моделі

            # Додавання додаткової інформації
            result_df['data_points'] = len(data)
            result_df['preparation_timestamp'] = pd.Timestamp.now()

            # Збереження в оригінальній карті даних
            self.original_data_map['arima_prepared_data'] = result_df.copy()

            self.logger.info(f"prepare_arima_data: Завершено підготовку даних для {symbol} ({timeframe})")

            return result_df

        except Exception as e:
            self.logger.error(f"Помилка при підготовці даних для ARIMA: {str(e)}")
            import traceback
            self.logger.error(f"Деталі помилки: {traceback.format_exc()}")
            return pd.DataFrame()



    def _determine_seasonal_period(self, timeframe: str) -> int:

            import re
            match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
            if not match:
                return 7  # Default to weekly seasonality

            value, unit = match.groups()
            value = int(value)

            # Set seasonal periods based on common patterns
            if unit.lower() in ['m', 'min', 'minute']:
                if value <= 5:
                    return 288  # Daily seasonality for 5m or less (288 5-min periods in a day)
                elif value <= 15:
                    return 96  # Daily seasonality for 15m (96 15-min periods in a day)
                elif value <= 60:
                    return 24  # Daily seasonality for hourly data (24 hours in a day)
            elif unit.lower() in ['h', 'hour']:
                return 24  # Daily seasonality for hourly data
            elif unit.lower() in ['d', 'day']:
                return 7  # Weekly seasonality for daily data
            elif unit.lower() in ['w', 'week']:
                return 4  # Monthly seasonality for weekly data

            # Default to weekly seasonality
            return 7

    def get_adaptive_sequence_length(self, timeframe: str) -> int:

        sequence_length_config = {
            '1m': (120, 240, 60),
            '1h': (72, 168, 24),
            '4h': (30, 90, 60),
            '1d': (45, 90, 30),
            '1w': (26, 52, 8)
        }

        default_value = (30, 90, 45)
        min_len, max_len, typical_len = sequence_length_config.get(timeframe, default_value)



        self.logger.info(f"Використовуємо адаптивний розмір sequence_length для {timeframe}: {typical_len} "
                         f"(можливий діапазон: {min_len}-{max_len})")

        return typical_len

    def prepare_lstm_data(
            self,
            data: pd.DataFrame | dd.DataFrame,
            symbol: str,
            timeframe: str,
            sequence_length: int = None,
            target_horizons: list = [1, 5, 10]
    ) -> pd.DataFrame:
        try:
            self.logger.info(f"Підготовка даних LSTM для збереження в базі даних: {symbol}, {timeframe}")

            if sequence_length is None:
                sequence_length = self.get_adaptive_sequence_length(timeframe)
                self.logger.info(f"Використання адаптивного sequence_length: {sequence_length} для {timeframe}")

            # Конвертуємо Dask DataFrame в pandas, якщо необхідно
            if hasattr(data, 'compute'):
                data = data.compute()

            if data.empty:
                self.logger.warning("Надано порожній DataFrame")
                return pd.DataFrame()

            # Додано діагностичну інформацію про індекс
            self.logger.info(f"Тип індексу: {type(data.index)}")
            if len(data.index) > 0:
                self.logger.info(f"Приклад першої дати: {data.index[0]} (тип: {type(data.index[0])})")
                if isinstance(data.index[0], pd.Timestamp):
                    self.logger.info(f"День тижня для першої дати: {data.index[0].day_name()}")

            # Переконуємось, що маємо DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.error("Дані повинні мати DatetimeIndex")
                # Спроба конвертації до DatetimeIndex
                try:
                    data.index = pd.DatetimeIndex(data.index)
                    self.logger.info("Успішно конвертовано індекс до DatetimeIndex")
                except Exception as e:
                    self.logger.error(f"Не вдалося конвертувати індекс: {str(e)}")
                    return pd.DataFrame()

            # ВАЖЛИВЕ ВИПРАВЛЕННЯ: Переконаймося, що дані відсортовані за часом
            data = data.sort_index()

            # 1. Вибираємо необхідні колонки ознак
            required_columns = ['open', 'high', 'low', 'close', 'volume']

            # Перевіряємо відсутні колонки (без врахування регістру)
            data_columns_lower = {col.lower(): col for col in data.columns}
            missing_cols = [col for col in required_columns if col.lower() not in data_columns_lower]

            if missing_cols:
                self.logger.error(f"Відсутні необхідні колонки: {missing_cols}")
                return pd.DataFrame()

            # Відображення на фактичні імена колонок в DataFrame
            feature_mapping = {col: data_columns_lower[col.lower()] for col in required_columns}

            # Створюємо копію зі стандартизованими іменами колонок
            df = data.copy()
            for std_name, actual_name in feature_mapping.items():
                if std_name != actual_name:
                    df[std_name] = df[actual_name]

            # 2. Додаємо часові ознаки з методу create_time_features
            df = self.create_time_features(df)

            # 3. Розраховуємо додаткові ознаки об'єму
            # Зміна об'єму (у відсотках)
            df['volume_change'] = df['volume'].pct_change(1)

            # Ковзне середнє об'єму (5 періодів)
            df['volume_rolling_mean'] = df['volume'].rolling(window=5).mean()

            # Ковзне стандартне відхилення об'єму (5 періодів)
            df['volume_rolling_std'] = df['volume'].rolling(window=5).std()

            # Стрибок об'єму (співвідношення поточного об'єму до ковзного середнього)
            df['volume_spike'] = df['volume'] / df['volume_rolling_mean']

            # Заповнюємо значення NaN нулями для розрахованих ознак
            volume_features = ['volume_change', 'volume_rolling_mean', 'volume_rolling_std', 'volume_spike']
            df[volume_features] = df[volume_features].fillna(0)

            # 4. Масштабуємо ознаки
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Вибираємо вибірку для налаштування скейлера, якщо набір даних дуже великий
            features_to_scale = required_columns + volume_features

            # Адаптивна стратегія для вибірки даних для скейлера залежно від таймфрейму
            scaling_samples = {
                '1m': 25000,
                '5m': 20000,
                '15m': 15000,
                '30m': 10000,
                '1h': 8000,
                '4h': 5000,
                '1d': 2500,
                '1w': 1000
            }

            scaling_sample_size = scaling_samples.get(timeframe, self.scaling_sample_size)

            if len(df) > scaling_sample_size:
                # Використовуємо стратифіковану вибірку для кращого представлення різних періодів
                time_periods = pd.qcut(np.arange(len(df)), 10, duplicates='drop')
                stratified_indices = []

                for period in time_periods.unique():
                    period_indices = np.where(time_periods == period)[0]
                    samples_per_period = min(len(period_indices), scaling_sample_size // 10)
                    period_sample = np.random.choice(period_indices, size=samples_per_period, replace=False)
                    stratified_indices.extend(period_sample)

                sample_df = df.iloc[stratified_indices][features_to_scale]
                scaler.fit(sample_df)
            else:
                scaler.fit(df[features_to_scale])

            # Масштабуємо необхідні колонки
            scaled_data = scaler.transform(df[features_to_scale])
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=[f"{col}_scaled" for col in features_to_scale],
                index=df.index
            )

            # Об'єднуємо з оригінальним DataFrame
            result_df = pd.concat([df, scaled_df], axis=1)

            # 5. Створюємо цільові значення для різних горизонтів
            for horizon in target_horizons:
                result_df[f'target_close_{horizon}'] = result_df['close'].shift(-horizon)

            # 6. Створюємо ідентифікатори послідовностей та позиції
            valid_end_idx = len(result_df) - max(target_horizons)

            # Визначаємо загальний розмір даних і адаптуємо параметри відповідно
            total_data_points = len(result_df)

            # ВАЖЛИВЕ ВИПРАВЛЕННЯ - Відстежуємо всі створені послідовності за датою початку
            sequence_start_times = {}

            # Оновлена конфігурація для всіх методів з установкою step = sequence_length
            timeframe_config = {
                '1m': {
                    'method': 'random_sample',
                    'sample_rate': self._adaptive_sample_rate(total_data_points, 0.001, 0.05, 500000),
                    'min_gap': sequence_length  # Встановлюємо мінімальний проміжок рівним sequence_length
                },
                '5m': {
                    'method': 'random_sample',
                    'sample_rate': self._adaptive_sample_rate(total_data_points, 0.005, 0.1, 200000),
                    'min_gap': sequence_length
                },
                '15m': {
                    'method': 'systematic_sample',
                    'sampling_fraction': self._adaptive_sample_rate(total_data_points, 0.02, 0.2, 100000),
                    'offset': 3,
                    'step': sequence_length  # Додаємо явний крок
                },
                '30m': {
                    'method': 'systematic_sample',
                    'sampling_fraction': self._adaptive_sample_rate(total_data_points, 0.05, 0.3, 50000),
                    'offset': 2,
                    'step': sequence_length  # Додаємо явний крок
                },
                '1h': {
                    'method': 'systematic_sample',
                    'sampling_fraction': self._adaptive_sample_rate(total_data_points, 0.1, 0.4, 20000),
                    'offset': 2,
                    'step': sequence_length  # Додаємо явний крок
                },
                '4h': {
                    'method': 'no_overlap',
                    'extra_gap': 0  # Встановлюємо extra_gap = 0 для відсутності додаткового проміжку
                },
                '1d': {
                    'method': 'sliding_window',  # Новий метод з перекриттям
                    'window_step': 1,            # Крок вікна (1 = максимальне перекриття)
                    'max_sequences': 4000       # Обмеження кількості послідовностей
                },
                '1w': {
                    'method': 'weekly_anchored',
                    'anchor_day': 0,  # Тільки понеділок (фіксовано)
                    'additional_days': [],  # Без додаткових днів для тижневих даних
                    'step': sequence_length  # Додаємо явний крок
                }
            }

            # За замовчуванням використовуємо метод без перекриття
            default_config = {
                'method': 'no_overlap',
                'extra_gap': 0  # Встановлюємо extra_gap = 0 для відсутності додаткового проміжку
            }

            # Отримуємо конфігурацію для даного часового інтервалу
            config = timeframe_config.get(timeframe, default_config)
            method = config.get('method')

            sequence_data = []

            self.logger.info(
                f"Використовується метод генерації послідовностей '{method}' для таймфрейму {timeframe} з {total_data_points} точками даних"
            )

            # Генерація послідовностей в залежності від методу
            if method == 'random_sample':
                # Випадковий відбір з мінімальними проміжками
                sample_rate = config.get('sample_rate', 0.1)
                min_gap = sequence_length  # ЗМІНА: Встановлюємо завжди рівним sequence_length

                # Логуємо актуальні параметри
                self.logger.info(f"Випадкова вибірка з частотою: {sample_rate}, мінімальний проміжок: {min_gap}")

                # Визначаємо всі допустимі початкові індекси
                valid_start_indices = list(
                    range(0, valid_end_idx - sequence_length, sequence_length))  # ЗМІНА: Крок sequence_length

                # Кількість послідовностей для відбору
                target_sequences = min(len(valid_start_indices), int(len(valid_start_indices) * sample_rate))
                target_sequences = min(target_sequences, 10000)  # Обмежуємо максимальну кількість послідовностей

                self.logger.info(f"Цільова кількість послідовностей для створення: {target_sequences}")

                # Відбираємо початкові точки без перекриття
                if len(valid_start_indices) > target_sequences:
                    # Обираємо випадкові початкові точки без перекриття
                    selected_indices = np.random.choice(valid_start_indices, size=target_sequences, replace=False)
                else:
                    selected_indices = valid_start_indices

                # Сортуємо індекси для збереження хронологічного порядку
                selected_indices = sorted(selected_indices)

                # Генеруємо послідовності з відібраних початкових точок
                self.logger.info(f"Фактично відібрано {len(selected_indices)} початкових точок послідовностей")

                # Створюємо на основі часу початку, а не індексу послідовності
                seq_counter = 0
                for start_idx in selected_indices:
                    start_time = result_df.index[start_idx]

                    # Перевіряємо, чи можемо створити повну послідовність
                    if start_idx + sequence_length > len(result_df) or start_idx + sequence_length + max(
                            target_horizons) > len(result_df):
                        continue

                    # Зберігаємо часову мітку початку для перевірки
                    sequence_start_times[seq_counter] = start_time

                    # Зберігаємо послідовності окремо для кожного seq_id
                    sequence_points = []
                    for pos in range(sequence_length):
                        idx = start_idx + pos
                        if 0 <= idx < len(result_df) and idx + max(target_horizons) < len(result_df):
                            sequence_point = self._create_sequence_data_point(result_df, idx, seq_counter, pos,
                                                                              timeframe, sequence_length,
                                                                              target_horizons,
                                                                              scaler, features_to_scale)
                            sequence_points.append(sequence_point)

                    # Додаємо послідовність тільки якщо вона повна
                    if len(sequence_points) == sequence_length:
                        sequence_data.extend(sequence_points)
                        seq_counter += 1
            elif method == 'sliding_window':
                # Ковзне вікно з заданим кроком
                window_step = config.get('window_step', 1)  # За замовчуванням крок = 1
                max_sequences = config.get('max_sequences', 4000)  # Обмеження на кількість послідовностей

                self.logger.info(
                    f"Використовується метод ковзного вікна з кроком: {window_step}, макс. послідовностей: {max_sequences}")

                seq_counter = 0
                # Створюємо послідовності з перекриттям
                for start_idx in range(0, valid_end_idx - sequence_length, window_step):
                    # Обмеження кількості послідовностей
                    if seq_counter >= max_sequences:
                        self.logger.info(f"Досягнуто максимальну кількість послідовностей ({max_sequences})")
                        break

                    start_time = result_df.index[start_idx]

                    # Перевіряємо, чи можемо створити повну послідовність
                    if start_idx + sequence_length > len(result_df) or start_idx + sequence_length + max(
                            target_horizons) > len(result_df):
                        continue

                    # Зберігаємо часову мітку початку для перевірки
                    sequence_start_times[seq_counter] = start_time

                    # Зберігаємо послідовності окремо для кожного seq_id
                    sequence_points = []
                    for pos in range(sequence_length):
                        idx = start_idx + pos
                        if 0 <= idx < len(result_df) and idx + max(target_horizons) < len(result_df):
                            sequence_point = self._create_sequence_data_point(result_df, idx, seq_counter, pos,
                                                                              timeframe, sequence_length,
                                                                              target_horizons,
                                                                              scaler, features_to_scale)
                            sequence_points.append(sequence_point)

                    # Додаємо послідовність тільки якщо вона повна
                    if len(sequence_points) == sequence_length:
                        sequence_data.extend(sequence_points)
                        seq_counter += 1
            elif method == 'systematic_sample':
                # Систематична вибірка з певним інтервалом
                step = sequence_length  # ЗМІНА: Крок завжди дорівнює sequence_length
                offset = config.get('offset', 0)

                self.logger.info(f"Систематична вибірка з кроком: {step}, зміщення: {offset}")

                seq_counter = 0
                # Створюємо на основі часу початку, а не індексу послідовності
                for start_idx in range(offset, valid_end_idx - sequence_length, step):
                    start_time = result_df.index[start_idx]

                    # Перевіряємо, чи можемо створити повну послідовність
                    if start_idx + sequence_length > len(result_df) or start_idx + sequence_length + max(
                            target_horizons) > len(result_df):
                        continue

                    # Зберігаємо часову мітку початку для перевірки
                    sequence_start_times[seq_counter] = start_time

                    # Зберігаємо послідовності окремо для кожного seq_id
                    sequence_points = []
                    for pos in range(sequence_length):
                        idx = start_idx + pos
                        if 0 <= idx < len(result_df) and idx + max(target_horizons) < len(result_df):
                            sequence_point = self._create_sequence_data_point(result_df, idx, seq_counter, pos,
                                                                              timeframe, sequence_length,
                                                                              target_horizons,
                                                                              scaler, features_to_scale)
                            sequence_points.append(sequence_point)

                    # Додаємо послідовність тільки якщо вона повна
                    if len(sequence_points) == sequence_length:
                        sequence_data.extend(sequence_points)
                        seq_counter += 1

            elif method == 'no_overlap':
                # Послідовності без перекриття
                extra_gap = 0  # ЗМІНА: Встановлюємо extra_gap = 0
                step = sequence_length + extra_gap  # Крок дорівнює sequence_length

                self.logger.info(f"Вибірка без перекриття з кроком: {step}")

                seq_counter = 0
                # Створюємо на основі часу початку, а не індексу послідовності
                for start_idx in range(0, valid_end_idx - sequence_length, step):
                    start_time = result_df.index[start_idx]

                    # Перевіряємо, чи можемо створити повну послідовність
                    if start_idx + sequence_length > len(result_df) or start_idx + sequence_length + max(
                            target_horizons) > len(result_df):
                        continue

                    # Зберігаємо часову мітку початку для перевірки
                    sequence_start_times[seq_counter] = start_time

                    # Зберігаємо послідовності окремо для кожного seq_id
                    sequence_points = []
                    for pos in range(sequence_length):
                        idx = start_idx + pos
                        if 0 <= idx < len(result_df) and idx + max(target_horizons) < len(result_df):
                            sequence_point = self._create_sequence_data_point(result_df, idx, seq_counter, pos,
                                                                              timeframe, sequence_length,
                                                                              target_horizons,
                                                                              scaler, features_to_scale)
                            sequence_points.append(sequence_point)

                    # Додаємо послідовність тільки якщо вона повна
                    if len(sequence_points) == sequence_length:
                        sequence_data.extend(sequence_points)
                        seq_counter += 1

            elif method == 'weekly_anchored':
                # Аналіз днів тижня в даних для діагностики
                day_counts = {}
                for i in range(len(result_df)):
                    dt = result_df.index[i].to_pydatetime()
                    day_counts[dt.weekday()] = day_counts.get(dt.weekday(), 0) + 1

                self.logger.info(f"Розподіл днів тижня в даних: {day_counts}")

                # Виведення перших кількох дат для аналізу
                if len(result_df) > 0:
                    first_dates = [result_df.index[i].strftime('%Y-%m-%d (%A)') for i in range(min(5, len(result_df)))]
                    self.logger.info(f"Перші 5 дат: {first_dates}")

                # Визначаємо step для неперекривних послідовностей
                step = sequence_length
                self.logger.info(f"Для таймфрейму використовуємо крок {step} (без перекриття)")

                if timeframe == '1w':
                    most_common_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else 0
                    if max(day_counts.values() if day_counts else [0]) > 5:
                        anchor_days = [most_common_day]
                        self.logger.info(
                            f"Використовуємо {['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'][most_common_day]} як день прив'язки")
                    else:
                        anchor_days = list(range(7))
                        self.logger.info("Недостатньо даних для одного дня тижня, використовуємо всі дні")
                else:
                    anchor_day = config.get('anchor_day', 0)
                    anchor_days = [anchor_day]  # ЗМІНА: Використовуємо тільки основний день прив'язки

                # Формуємо anchor_dates з кроком sequence_length
                anchor_dates = []
                for i in range(0, valid_end_idx - sequence_length + 1, step):
                    dt = result_df.index[i].to_pydatetime()
                    if dt.weekday() in anchor_days:
                        anchor_dates.append(i)

                self.logger.info(f"Знайдено {len(anchor_dates)} дат прив'язки")

                # Якщо все ще немає дат прив'язки, використовуємо альтернативний підхід
                if not anchor_dates:
                    self.logger.warning("Не знайдено відповідних дат прив'язки, використовуємо метод без перекриття")
                    # Використовуємо метод без перекриття як резервний
                    for start_idx in range(0, valid_end_idx - sequence_length, sequence_length):
                        if start_idx + sequence_length <= valid_end_idx:
                            anchor_dates.append(start_idx)

                # Сортуємо індекси для збереження хронологічного порядку
                anchor_dates = sorted(anchor_dates)

                # Створюємо на основі часу початку, а не індексу послідовності
                seq_counter = 0
                for start_idx in anchor_dates:
                    # Перевіряємо, чи можемо створити повну послідовність
                    if start_idx + sequence_length > len(result_df) or start_idx + sequence_length + max(
                            target_horizons) > len(result_df):
                        continue

                    start_time = result_df.index[start_idx]

                    # Зберігаємо часову мітку початку для перевірки
                    sequence_start_times[seq_counter] = start_time

                    # Зберігаємо послідовності окремо для кожного seq_id
                    sequence_points = []
                    for pos in range(sequence_length):
                        idx = start_idx + pos
                        if 0 <= idx < len(result_df) and idx + max(target_horizons) < len(result_df):
                            sequence_point = self._create_sequence_data_point(result_df, idx, seq_counter, pos,
                                                                              timeframe, sequence_length,
                                                                              target_horizons,
                                                                              scaler, features_to_scale)
                            sequence_points.append(sequence_point)

                    # Додаємо послідовність тільки якщо вона повна
                    if len(sequence_points) == sequence_length:
                        sequence_data.extend(sequence_points)
                        seq_counter += 1

            elif method == 'monthly_anchored':
                # Прив'язка до початку місяця з вибором певних точок
                # ЗМІНА: Модифікуємо для забезпечення відсутності перекриття
                self.logger.info(f"Місячна прив'язка з кроком {sequence_length}")

                # Групуємо дати за місяцями
                month_groups = {}

                for i in range(0, valid_end_idx - sequence_length + 1,
                               sequence_length):  # ЗМІНА: Використовуємо крок sequence_length
                    dt = result_df.index[i].to_pydatetime()
                    month_key = (dt.year, dt.month)

                    if month_key not in month_groups:
                        month_groups[month_key] = []

                    month_groups[month_key].append(i)

                self.logger.info(f"Знайдено дані для {len(month_groups)} місяців")

                # Сортуємо місяці для послідовної обробки в хронологічному порядку
                sorted_months = sorted(month_groups.keys())

                # Вибираємо точки з перевагою для понеділків без перекриття
                seq_counter = 0

                for month in sorted_months:
                    indices = month_groups[month]

                    # Пропускаємо місяці з недостатньою кількістю даних
                    if len(indices) < sequence_length:
                        self.logger.warning(
                            f"Пропускаємо місяць {month} з недостатньою кількістю даних ({len(indices)} точок)")
                        continue

                    # Визначаємо всі понеділки в цьому місяці (бажані початкові точки)
                    monday_indices = []

                    for idx in indices:
                        dt = result_df.index[idx].to_pydatetime()
                        if dt.weekday() == 0:  # Понеділок
                            monday_indices.append(idx)

                    selected_indices = []

                    # Якщо у нас є понеділки, пріоритетизуємо їх
                    if monday_indices:
                        # Сортуємо понеділки за днем місяця
                        monday_indices.sort(key=lambda idx: result_df.index[idx].day)

                        # ЗМІНА: Вибираємо понеділки з кроком, що забезпечує відсутність перекриття
                        current_idx = 0
                        while current_idx < len(monday_indices):
                            selected_indices.append(monday_indices[current_idx])

                            # Визначаємо наступний індекс, який не перекривається з поточним
                            next_idx = current_idx + 1
                            while next_idx < len(monday_indices) and monday_indices[next_idx] < monday_indices[
                                current_idx] + sequence_length:
                                next_idx += 1

                            current_idx = next_idx

                    # Якщо у нас немає жодного понеділка, використовуємо перший індекс місяця
                    if not selected_indices and indices:
                        selected_indices = [indices[0]]

                    # Створюємо на основі часу початку, а не індексу послідовності
                    for start_idx in selected_indices:
                        # Перевіряємо, чи можемо створити повну послідовність
                        if start_idx + sequence_length > len(result_df) or start_idx + sequence_length + max(
                                target_horizons) > len(result_df):
                            continue

                        start_time = result_df.index[start_idx]

                        # Зберігаємо часову мітку початку для перевірки
                        sequence_start_times[seq_counter] = start_time

                        sequence_points = []
                        for pos in range(sequence_length):
                            idx = start_idx + pos
                            if 0 <= idx < len(result_df) and idx + max(target_horizons) < len(result_df):
                                sequence_point = self._create_sequence_data_point(result_df, idx, seq_counter, pos,
                                                                                  timeframe, sequence_length,
                                                                                  target_horizons,
                                                                                  scaler, features_to_scale)
                                sequence_points.append(sequence_point)

                        # Додаємо послідовність тільки якщо вона повна
                        if len(sequence_points) == sequence_length:
                            sequence_data.extend(sequence_points)
                            seq_counter += 1
            else:
                # За замовчуванням - використовуємо крок sequence_length без перекриття
                step = sequence_length  # ЗМІНА: Завжди використовуємо sequence_length

                self.logger.info(f"Вибірка за замовчуванням з кроком: {step}")

                seq_counter = 0
                # Створюємо на основі часу початку, а не індексу послідовності
                for start_idx in range(0, valid_end_idx - sequence_length, step):
                    start_time = result_df.index[start_idx]

                    # Перевіряємо, чи можемо створити повну послідовність
                    if start_idx + sequence_length > len(result_df) or start_idx + sequence_length + max(
                            target_horizons) > len(result_df):
                        continue

                    # Зберігаємо часову мітку початку для перевірки
                    sequence_start_times[seq_counter] = start_time

                    # Зберігаємо послідовності окремо для кожного seq_id
                    sequence_points = []
                    for pos in range(sequence_length):
                        idx = start_idx + pos
                        if 0 <= idx < len(result_df) and idx + max(target_horizons) < len(result_df):
                            sequence_point = self._create_sequence_data_point(result_df, idx, seq_counter, pos,
                                                                              timeframe, sequence_length,
                                                                              target_horizons,
                                                                              scaler, features_to_scale)
                            sequence_points.append(sequence_point)

                    # Додаємо послідовність тільки якщо вона повна
                    if len(sequence_points) == sequence_length:
                        sequence_data.extend(sequence_points)
                        seq_counter += 1

            # Перевіряємо, чи створені послідовності
            if not sequence_data:
                self.logger.warning(f"Не створено жодної послідовності для {symbol}, {timeframe}")
                return pd.DataFrame()

            # Створюємо фінальний DataFrame з послідовностями
            sequences_df = pd.DataFrame(sequence_data)

            # Логування статистики створених послідовностей
            num_sequences = len(sequence_start_times)
            self.logger.info(f"Створено {num_sequences} послідовностей для {symbol}, {timeframe}")
            self.logger.info(f"Загальна кількість точок даних: {len(sequences_df)}")

            # Додаємо метадані про скейлер для використання при інференсі
            sequences_df.attrs['scaler'] = scaler
            sequences_df.attrs['features_to_scale'] = features_to_scale
            sequences_df.attrs['symbol'] = symbol
            sequences_df.attrs['timeframe'] = timeframe
            sequences_df.attrs['sequence_length'] = sequence_length
            sequences_df.attrs['target_horizons'] = target_horizons

            return sequences_df

        except Exception as e:
            self.logger.error(f"Error in prepare_lstm_data: {str(e)}", exc_info=True)
            return pd.DataFrame()


    def _create_sequence_data_point(self, result_df, idx, seq_id, pos,
                                        timeframe, sequence_length, target_horizons,
                                        scaler, features_to_scale):
            """
            Створює точку даних для послідовності з перевіркою валідності індексу
            """
            row = result_df.iloc[idx].copy()
            return {
                'timeframe': timeframe,
                'sequence_id': seq_id,
                'sequence_position': pos,
                'open_time': result_df.index[idx],

                # Scaled features
                'open_scaled': float(row['open_scaled']),
                'high_scaled': float(row['high_scaled']),
                'low_scaled': float(row['low_scaled']),
                'close_scaled': float(row['close_scaled']),
                'volume_scaled': float(row['volume_scaled']),

                # Added volume features
                'volume_change_scaled': float(row['volume_change_scaled']),
                'volume_rolling_mean_scaled': float(row['volume_rolling_mean_scaled']),
                'volume_rolling_std_scaled': float(row['volume_rolling_std_scaled']),
                'volume_spike_scaled': float(row['volume_spike_scaled']),

                # Time features
                'hour_sin': float(row['hour_sin']),
                'hour_cos': float(row['hour_cos']),
                'day_of_week_sin': float(row['day_of_week_sin']),
                'day_of_week_cos': float(row['day_of_week_cos']),
                'month_sin': float(row['month_sin']),
                'month_cos': float(row['month_cos']),
                'day_of_month_sin': float(row['day_of_month_sin']),
                'day_of_month_cos': float(row['day_of_month_cos']),

                # Target values
                'target_close_1': float(row['target_close_1']) if 1 in target_horizons else None,
                'target_close_5': float(row['target_close_5']) if 5 in target_horizons else None,
                'target_close_10': float(row['target_close_10']) if 10 in target_horizons else None,

                # Metadata
                'sequence_length': sequence_length,
                'scaling_metadata': json.dumps({
                    'feature_range': scaler.feature_range,
                    'data_min': scaler.data_min_.tolist(),
                    'data_max': scaler.data_max_.tolist(),
                    'columns': features_to_scale
                })
            }

    def _adaptive_sample_rate(self, total_points, min_rate=0.001, max_rate=0.1, threshold=500000):
        if total_points >= threshold:
            return min_rate
        elif total_points <= threshold / 10:
            return max_rate
        else:
            # Лінійна інтерполяція між min_rate і max_rate
            factor = (threshold - total_points) / (threshold - threshold / 10)
            return min_rate + factor * (max_rate - min_rate)

    def validate_sequence_integrity(self, final_df: pd.DataFrame, timeframe: str, sequence_length: int):

        self.logger.info(f"Validating sequence integrity for {timeframe} data...")

        if final_df.empty:
            self.logger.warning("Empty DataFrame - nothing to validate")
            return final_df

        if 'sequence_id' not in final_df.columns:
            self.logger.error("Missing sequence_id column - cannot validate")
            return final_df

        # Count rows per sequence
        sequence_counts = final_df['sequence_id'].value_counts()

        # Identify incomplete sequences
        incomplete_sequences = sequence_counts[sequence_counts != sequence_length]

        if not incomplete_sequences.empty:
            self.logger.warning(f"Found {len(incomplete_sequences)} incomplete sequences")

            # Detailed logging for the first few incomplete sequences
            for i, (seq_id, count) in enumerate(incomplete_sequences.items()[:5]):
                sequence_data = final_df[final_df['sequence_id'] == seq_id]
                seq_times = sequence_data['open_time'].sort_values()

                self.logger.warning(f"Example incomplete sequence {seq_id}: {count}/{sequence_length} points")
                self.logger.warning(f"First time: {seq_times.iloc[0]}, Last time: {seq_times.iloc[-1]}")

                if 'sequence_position' in sequence_data.columns:
                    missing_positions = set(range(sequence_length)) - set(sequence_data['sequence_position'])
                    self.logger.warning(f"Missing positions: {sorted(missing_positions)}")

            # Remove incomplete sequences
            self.logger.info(f"Removing {len(incomplete_sequences)} incomplete sequences")
            valid_sequences = sequence_counts[sequence_counts == sequence_length].index
            final_df = final_df[final_df['sequence_id'].isin(valid_sequences)]

        # Check for weekday consistency in weekly data
        if timeframe == '1w':
            weekday_starts = {}
            for seq_id in final_df['sequence_id'].unique():
                seq_data = final_df[final_df['sequence_id'] == seq_id]
                first_row = seq_data[seq_data['sequence_position'] == 0]

                if not first_row.empty:
                    start_time = first_row['open_time'].iloc[0]
                    weekday = start_time.to_pydatetime().weekday()

                    if weekday not in weekday_starts:
                        weekday_starts[weekday] = 0
                    weekday_starts[weekday] += 1

            weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_summary = {weekday_names[day]: count for day, count in weekday_starts.items()}

            self.logger.info(f"Weekly sequence start day distribution: {weekday_summary}")

            # Alert if we don't have consistent Monday starts
            if 0 not in weekday_starts or weekday_starts[0] < sum(weekday_starts.values()) * 0.9:
                self.logger.warning(
                    f"Weekly data should consistently start on Mondays, but found: {weekday_summary}"
                )

        return final_df