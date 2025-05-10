import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
import pytz
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from data_collection.AnomalyDetector import AnomalyDetector
from data_collection.DataResampler import DataResampler
from utils.config import BINANCE_API_KEY, BINANCE_API_SECRET


class DataCleaner:
    def __init__(self, logger=None):
        self.logger = logger if logger else self._setup_default_logger()
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        self.data_resampler = DataResampler(logger=self.logger)

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        logger = logging.getLogger("data_cleaner")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _fix_invalid_high_low(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Виправляє некоректні співвідношення між high і low значеннями.
        Враховує особливість криптовалютних даних, де high та low можуть бути рівними.
        """
        result = data.copy()

        # Перевірка на наявність необхідних колонок
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
        if len(price_cols) < 4:
            return result

        # 1. Перевірка та виправлення випадків high < low
        invalid_hl = result['high'] < result['low']
        if invalid_hl.any():
            invalid_count = invalid_hl.sum()
            self.logger.warning(f"Знайдено {invalid_count} рядків, де high < low")

            if self.logger.isEnabledFor(logging.DEBUG):
                invalid_indexes = result.index[invalid_hl].tolist()
                self.logger.debug(
                    f"Індекси проблемних рядків: {invalid_indexes[:10]}{'...' if len(invalid_indexes) > 10 else ''}")

            # Міняємо місцями high та low значення
            temp = result.loc[invalid_hl, 'high'].copy()
            result.loc[invalid_hl, 'high'] = result.loc[invalid_hl, 'low']
            result.loc[invalid_hl, 'low'] = temp

        # 2. Додаткова перевірка на випадки, коли high = low
        equal_hl = (result['high'] == result['low']) & (result['volume'] > 0)
        if equal_hl.any():
            equal_count = equal_hl.sum()
            self.logger.info(f"Знайдено {equal_count} рядків з рівними high і low при ненульовому об'ємі")

            # Для таких записів можна додати невелику різницю, але лише якщо це необхідно для подальшої обробки
            # Зазвичай рівні high/low в криптовалютах - це нормальна ситуація

        # 3. Додаткова перевірка на випадки, коли всі OHLC значення рівні (flat price)
        flat_price = (result['open'] == result['high']) & (result['high'] == result['low']) & (
                    result['low'] == result['close']) & (result['volume'] > 0)
        if flat_price.any():
            flat_count = flat_price.sum()
            self.logger.info(f"Знайдено {flat_count} записів з однаковими OHLC значеннями при ненульовому об'ємі")
            # Це нормальна ситуація для неліквідних періодів на криптовалютному ринку

        return result

    def _remove_outliers(self, data: pd.DataFrame, std_dev: float = 3.0,
                         price_pct_threshold: float = 0.5) -> pd.DataFrame:
        """
        Видаляє аномальні значення, враховуючи високу волатильність криптовалютного ринку.

        Args:
            data: DataFrame з даними
            std_dev: Множник для IQR при визначенні викидів
            price_pct_threshold: Поріг для % зміни ціни як додаткова перевірка (50% за замовчуванням)

        Returns:
            DataFrame без аномальних значень
        """
        self.logger.info("Видалення аномальних значень...")
        result = data.copy()

        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]

        # Створюємо колонку для визначення цінових стрибків (price jumps)
        if len(data) > 1 and 'close' in result.columns:
            result['price_pct_change'] = result['close'].pct_change().abs()

        for col in price_cols:
            # Перевірка на порожню колонку
            if col not in result.columns or result[col].empty or result[col].isna().all():
                continue

            # Використання IQR для виявлення викидів
            Q1 = result[col].quantile(0.25)
            Q3 = result[col].quantile(0.75)
            IQR = Q3 - Q1

            # Для криптовалют розширюємо діапазон допустимих значень через високу волатильність
            lower_bound = Q1 - std_dev * IQR
            upper_bound = Q3 + std_dev * IQR

            # Перевірка на негативні значення нижньої межі (для криптовалют неможливі негативні ціни)
            if lower_bound < 0:
                lower_bound = 0
                self.logger.info(f"Скоригована нижня межа для {col} до 0 (замість негативного значення)")

            # Основний фільтр на викиди по IQR
            outliers = (result[col] < lower_bound) | (result[col] > upper_bound)

            # Додаткова перевірка для різких стрибків ціни (якщо був створений price_pct_change)
            if 'price_pct_change' in result.columns:
                # Виключаємо з викидів різкі стрибки, які можуть бути легітимними для криптовалют
                # при значних новинах чи ринкових подіях
                price_jumps = result['price_pct_change'] > price_pct_threshold
                self.logger.info(
                    f"Знайдено {price_jumps.sum()} значних цінових стрибків (>{price_pct_threshold * 100}%)")

                # Не враховуємо цінові стрибки як викиди, якщо вони не занадто екстремальні
                extreme_price_jumps = result['price_pct_change'] > price_pct_threshold * 2
                outliers = outliers & (~price_jumps | extreme_price_jumps)

            if outliers.any():
                outlier_count = outliers.sum()
                outlier_indexes = result.index[outliers].tolist()
                self.logger.info(f"Знайдено {outlier_count} аномалій в колонці {col}")

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"Індекси перших 10 аномалій: {outlier_indexes[:10]}{'...' if len(outlier_indexes) > 10 else ''}")

                # Замінюємо викиди на NaN для подальшого заповнення
                result.loc[outliers, col] = np.nan

        # Видаляємо допоміжну колонку
        if 'price_pct_change' in result.columns:
            result.drop('price_pct_change', axis=1, inplace=True)

        return result

    def _fix_invalid_values(self, data: pd.DataFrame, essential_cols: List[str]) -> pd.DataFrame:
        """Виправляє неприпустимі значення: від'ємні ціни, нульові high/low та аномальні об'єми"""
        result = data.copy()

        # Перевірка на від'ємні ціни
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
        for col in price_cols:
            negative_prices = result[col] < 0
            if negative_prices.any():
                negative_count = negative_prices.sum()
                self.logger.warning(f"Знайдено {negative_count} від'ємних значень у колонці {col}")

                # Замість використання абсолютних значень, позначаємо ці дані як проблемні та заміняємо їх медіаною колонки
                if result[col].median() > 0:  # Переконуємося, що медіана не є від'ємною або нульовою
                    result.loc[negative_prices, col] = result[col].median()
                    self.logger.info(
                        f"Від'ємні значення у колонці {col} замінені на медіану колонки: {result[col].median()}")
                else:
                    # Якщо медіана теж некоректна, використовуємо останнє коректне значення або невелике додатне число
                    result.loc[negative_prices, col] = result[col].abs().mean() or 0.0001
                    self.logger.info(
                        f"Від'ємні значення у колонці {col} замінені на середнє абсолютних значень: {result[col].abs().mean() or 0.0001}")

            # Перевірка на нульові значення high/low, що нетипово для криптовалют
            zero_prices = result[col] == 0
            if zero_prices.any() and col in ['high', 'low']:
                zero_count = zero_prices.sum()
                self.logger.warning(f"Знайдено {zero_count} нульових значень у колонці {col}")

                # Для high використовуємо максимум із open та close
                if col == 'high':
                    # Перевіряємо, що хоча б одне значення не нульове
                    mask = (result.loc[zero_prices, 'open'] > 0) | (result.loc[zero_prices, 'close'] > 0)
                    result.loc[zero_prices & mask, col] = result.loc[zero_prices & mask, ['open', 'close']].max(axis=1)

                    # Якщо обидва значення нульові, використовуємо медіану або мінімальне ненульове значення колонки
                    still_zero = zero_prices & ~mask
                    if still_zero.any():
                        if (result[col] > 0).any():
                            result.loc[still_zero, col] = result[result[col] > 0][col].median()
                        else:
                            result.loc[still_zero, col] = 0.0001  # Мінімальне додатне значення як запасний варіант
                        self.logger.info(
                            f"Нульові значення в колонці {col} з нульовими open/close замінені на медіану або мінімальне додатне значення")

                # Для low використовуємо мінімум із open та close
                elif col == 'low':
                    # Перевіряємо, що хоча б одне значення не нульове
                    mask = (result.loc[zero_prices, 'open'] > 0) | (result.loc[zero_prices, 'close'] > 0)
                    result.loc[zero_prices & mask, col] = result.loc[zero_prices & mask, ['open', 'close']].min(axis=1)

                    # Якщо обидва значення нульові, використовуємо медіану або мінімальне ненульове значення колонки
                    still_zero = zero_prices & ~mask
                    if still_zero.any():
                        if (result[col] > 0).any():
                            result.loc[still_zero, col] = result[result[col] > 0][col].median()
                        else:
                            result.loc[still_zero, col] = 0.0001  # Мінімальне додатне значення як запасний варіант
                        self.logger.info(
                            f"Нульові значення в колонці {col} з нульовими open/close замінені на медіану або мінімальне додатне значення")

                self.logger.info(f"Нульові значення у колонці {col} виправлені")

        # Перевірка на від'ємні об'єми
        if 'volume' in result.columns:
            negative_volumes = result['volume'] < 0
            if negative_volumes.any():
                negative_count = negative_volumes.sum()
                self.logger.warning(f"Знайдено {negative_count} від'ємних значень об'єму")

                # Замінюємо від'ємні значення на медіану або невелике додатне значення
                if result['volume'].median() > 0:
                    result.loc[negative_volumes, 'volume'] = result['volume'].median()
                    self.logger.info(f"Від'ємні значення об'єму замінені на медіану: {result['volume'].median()}")
                else:
                    result.loc[negative_volumes, 'volume'] = result['volume'].abs().mean() or 0.01
                    self.logger.info(
                        f"Від'ємні значення об'єму замінені на середнє абсолютних значень: {result['volume'].abs().mean() or 0.01}")

            # Перевірка на нульові об'єми
            zero_volumes = result['volume'] == 0
            if zero_volumes.any():
                zero_count = zero_volumes.sum()
                self.logger.warning(f"Знайдено {zero_count} нульових значень об'єму")

                # Замінюємо нульові значення мінімальним ненульовим об'ємом або невеликим додатним значенням
                if (result['volume'] > 0).any():
                    min_volume = result[result['volume'] > 0]['volume'].min()
                    result.loc[zero_volumes, 'volume'] = min_volume
                    self.logger.info(f"Нульові значення об'єму замінені на мінімальний ненульовий об'єм: {min_volume}")
                else:
                    result.loc[zero_volumes, 'volume'] = 0.01
                    self.logger.info("Нульові значення об'єму замінені на 0.01")

            # Перевірка на аномально великі об'єми (можливі помилки даних)
            if len(result) > 10:  # потрібно достатньо даних для розрахунку
                volume_mean = result['volume'].mean()
                volume_std = result['volume'].std()
                abnormal_volume = result['volume'] > (volume_mean + 10 * volume_std)  # 10 стандартних відхилень

                if abnormal_volume.any():
                    abnormal_count = abnormal_volume.sum()
                    self.logger.warning(f"Знайдено {abnormal_count} аномально великих значень об'єму")

                    # Відмічаємо їх для додаткової перевірки, але не змінюємо автоматично
                    # Криптовалютні ринки можуть мати легітимні великі скачки об'єму при значних подіях
                    if self.logger.isEnabledFor(logging.DEBUG):
                        abnormal_indexes = result.index[abnormal_volume].tolist()
                        self.logger.debug(
                            f"Індекси аномальних об'ємів: {abnormal_indexes[:10]}{'...' if len(abnormal_indexes) > 10 else ''}")

        # Перевірка логічного відношення значень high/low/open/close
        if all(col in result.columns for col in ['high', 'low', 'open', 'close']):
            # Перевіряємо, що high завжди найбільше значення
            invalid_high = result['high'] < result[['open', 'close', 'low']].max(axis=1)
            if invalid_high.any():
                self.logger.warning(f"Знайдено {invalid_high.sum()} записів, де high менше за open, close або low")
                result.loc[invalid_high, 'high'] = result.loc[invalid_high, ['open', 'close', 'low']].max(axis=1)
                self.logger.info("Виправлено невалідні значення high")

            # Перевіряємо, що low завжди найменше значення
            invalid_low = result['low'] > result[['open', 'close']].min(axis=1)
            if invalid_low.any():
                self.logger.warning(f"Знайдено {invalid_low.sum()} записів, де low більше за open або close")
                result.loc[invalid_low, 'low'] = result.loc[invalid_low, ['open', 'close']].min(axis=1)
                self.logger.info("Виправлено невалідні значення low")

        return result

    def add_crypto_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для додавання крипто-специфічних ознак")
            return data if data is not None else pd.DataFrame()

        result = data.copy()

        # Перевіряємо наявність необхідних колонок
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in result.columns]
        if missing_cols:
            self.logger.warning(f"Відсутні необхідні колонки для розрахунку крипто-ознак: {missing_cols}")
            return result

        try:
            # 1. Волатильність
            result['volatility'] = (result['high'] - result['low']) / result['low']

            # 2. Об'єм/волатильність співвідношення (індикатор ліквідності)
            result['volume_volatility_ratio'] = result['volume'] / (
                        result['volatility'] + 0.0001)  # уникаємо ділення на нуль

            # 3. Індикатор нульового об'єму (характерно для неактивних періодів)
            result['zero_volume'] = (result['volume'] == 0).astype(int)

            # 4. Flat price індикатор (всі ціни однакові)
            result['flat_price'] = ((result['open'] == result['high']) &
                                    (result['high'] == result['low']) &
                                    (result['low'] == result['close'])).astype(int)

            # 5. Відсоткова зміна ціни
            result['price_change_pct'] = (result['close'] - result['open']) / result['open']

            # 6. Candle body size (відносний)
            result['body_size_pct'] = abs(result['close'] - result['open']) / result['open']

            # 7. Upper and lower shadows (відносні)
            result['upper_shadow_pct'] = (result['high'] - result[['open', 'close']].max(axis=1)) / result['open']
            result['lower_shadow_pct'] = (result[['open', 'close']].min(axis=1) - result['low']) / result['open']

            # 8. Is Doji (тіло свічки дуже мале)
            doji_threshold = 0.0005  # 0.05%
            result['is_doji'] = (result['body_size_pct'] < doji_threshold).astype(int)

            # 9. Volume spike (раптовий скачок об'єму) - використовуємо ковзне середнє
            if len(result) > 24:  # потрібно достатньо історії
                vol_ma = result['volume'].rolling(window=24).mean()
                result['volume_spike'] = (result['volume'] > vol_ma * 3).astype(int)

            self.logger.info(f"Успішно додано криптовалютні ознаки")

        except Exception as e:
            self.logger.error(f"Помилка при додаванні криптовалютних ознак: {str(e)}")

        return result

    def clean_data(self,
                   data: pd.DataFrame,
                   remove_outliers: bool = True,
                   fill_missing: bool = True,
                   normalize: bool = True,
                   norm_method: str = 'z-score',
                   resample: bool = True,
                   target_interval: str = None,
                   add_time_features: bool = True,
                   cyclical: bool = True,
                   add_sessions: bool = False,
                   add_crypto_features: bool = False,  # Новий параметр для крипто-специфічних ознак
                   crypto_volatility_tolerance: float = 0.5) -> pd.DataFrame:  # Параметр для врахування волатильності криптовалют

        self.logger.info(f"Початок комплексного очищення даних для криптовалютних таймсерій")

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для очищення")
            return data if data is not None else pd.DataFrame()

        # 1. Підготовка та валідація даних
        result = self._prepare_dataframe(data)

        # Збережемо копію оригінальних даних для відновлення при необхідності
        original_data = result.copy()

        # Список обов'язкових колонок
        essential_cols = ['open', 'high', 'low', 'close', 'volume']

        # 2. Перевірка наявності обов'язкових колонок перед початком обробки
        missing_essential = [col for col in essential_cols if col not in result.columns]
        if missing_essential:
            self.logger.error(f"Відсутні обов'язкові колонки на початку: {missing_essential}")
            raise ValueError(f"Вхідні дані не містять обов'язкових колонок: {missing_essential}")

        # 3. Виправлення неприпустимих значень (від'ємні ціни та об'єми)
        result = self._fix_invalid_values(result, essential_cols)
        self._verify_essential_columns(result, essential_cols)

        # 4. Видалення дублікатів
        result = self._remove_duplicates(result)
        self._verify_essential_columns(result, essential_cols)

        # 5. Видалення викидів (з врахуванням підвищеної волатильності криптовалют)
        if remove_outliers:
            result = self._remove_outliers(result, std_dev=5.0, price_pct_threshold=crypto_volatility_tolerance)
            self._verify_essential_columns(result, essential_cols)

        # 6. Заповнення відсутніх значень
        if fill_missing:
            result = self.handle_missing_values(result)
            self._verify_essential_columns(result, essential_cols)

        # 7. Виправлення невалідних high/low значень
        result = self._fix_invalid_high_low(result)
        self._verify_essential_columns(result, essential_cols)

        # 8. Ресемплінг даних
        if resample and target_interval and isinstance(result.index, pd.DatetimeIndex):
            backup_data = result.copy()  # Створюємо резервну копію перед ресемплінгом
            try:
                result = self.data_resampler.resample_data(result, target_interval)
                # Перевірка на пустий результат або відсутність необхідних колонок
                if result.empty or any(col not in result.columns for col in essential_cols):
                    self.logger.error("Ресемплінг призвів до втрати даних або колонок. Відновлення з резервної копії.")
                    result = backup_data
                else:
                    self._verify_essential_columns(result, essential_cols)
            except Exception as e:
                self.logger.error(f"Помилка під час ресемплінгу: {str(e)}. Відновлення з резервної копії.")
                result = backup_data

        # 9. Нормалізація даних (перед додаванням часових ознак)
        if normalize:
            # Зберігаємо копію перед нормалізацією
            pre_normalize_data = result.copy()

            # Виконуємо нормалізацію тільки для необхідних колонок
            try:
                normalized_result, scaler_meta = self.normalize_data(
                    result,
                    norm_method,
                    columns=essential_cols
                )

                if normalized_result is not None:
                    # Перевіряємо результати нормалізації
                    is_valid = self._validate_normalized_data(normalized_result, essential_cols)

                    if is_valid:
                        result = normalized_result
                        self._verify_essential_columns(result, essential_cols)
                    else:
                        self.logger.warning("Результати нормалізації недійсні. Використання оригінальних даних.")
                        result = pre_normalize_data
                else:
                    self.logger.warning("Нормалізація не вдалася, використовуємо оригінальні дані")
                    result = pre_normalize_data
            except Exception as e:
                self.logger.error(f"Помилка під час нормалізації: {str(e)}. Використання оригінальних даних.")
                result = pre_normalize_data

        # 10. Додавання часових ознак (після нормалізації)
        if add_time_features and isinstance(result.index, pd.DatetimeIndex):
            # Зберігаємо копію перед додаванням часових ознак
            pre_time_features_data = result.copy()

            try:
                # Додаємо часові ознаки безпечно, без перезапису існуючих колонок
                result = self.add_time_features_safely(result, cyclical, add_sessions)

                # Перевіряємо, чи всі обов'язкові колонки збереглися
                missing_after_features = [col for col in essential_cols if col not in result.columns]
                if missing_after_features:
                    self.logger.error(f"Втрачено колонки після додавання часових ознак: {missing_after_features}")
                    # Відновлюємо з попередньої копії
                    result = pre_time_features_data
                    self.logger.info("Відновлено дані з резервної копії")
            except Exception as e:
                self.logger.error(f"Помилка при додаванні часових ознак: {str(e)}. Використання попередніх даних.")
                result = pre_time_features_data

        # 11. NEW: Додавання крипто-специфічних ознак
        if add_crypto_features:
            pre_crypto_features_data = result.copy()

            try:
                result = self.add_crypto_specific_features(result)

                # Перевіряємо, чи всі обов'язкові колонки збереглися
                missing_after_crypto = [col for col in essential_cols if col not in result.columns]
                if missing_after_crypto:
                    self.logger.error(f"Втрачено колонки після додавання крипто-ознак: {missing_after_crypto}")
                    result = pre_crypto_features_data
                    self.logger.info("Відновлено дані з резервної копії")
            except Exception as e:
                self.logger.error(f"Помилка при додаванні крипто-ознак: {str(e)}. Використання попередніх даних.")
                result = pre_crypto_features_data

        # Фінальна перевірка
        self._verify_essential_columns(result, essential_cols)
        issues = self.validate_data_integrity(result)

        if issues:
            issue_count = sum(1 for issue in issues.values() if issue)
            self.logger.warning(f"Після очищення залишилось {issue_count} проблем з цілісністю даних")
        else:
            self.logger.info("Дані успішно очищені, проблем з цілісністю не виявлено")

        self.logger.info(f"Очищення даних завершено: {result.shape[0]} рядків, {result.shape[1]} стовпців")
        return result



    def _validate_normalized_data(self, data: pd.DataFrame, essential_cols: List[str]) -> bool:
        """Перевіряє коректність нормалізованих даних"""
        # Перевірка на NaN
        for col in essential_cols:
            if col not in data.columns:
                self.logger.error(f"Колонка {col} відсутня після нормалізації")
                return False

            if data[col].isna().any():
                self.logger.error(f"Виявлено NaN значення в нормалізованій колонці {col}")
                return False

            # Перевірка на нескінченні значення
            if np.isinf(data[col]).any():
                self.logger.error(f"Виявлено нескінченні значення в нормалізованій колонці {col}")
                return False

        # Перевірка співвідношення high і low
        if 'high' in data.columns and 'low' in data.columns:
            invalid_hl = data['high'] < data['low']
            if invalid_hl.any():
                self.logger.warning(f"Після нормалізації виявлено {invalid_hl.sum()} рядків з high < low")

        return True

    # Нова допоміжна функція для перевірки наявності необхідних колонок
    def _verify_essential_columns(self, data: pd.DataFrame, essential_cols: List[str]):
        missing_essential = [col for col in essential_cols if col not in data.columns]
        if missing_essential:
            self.logger.error(f"КРИТИЧНА ПОМИЛКА: Відсутні важливі колонки: {missing_essential}")
            raise ValueError(f"Втрачено критичні колонки: {missing_essential}")
        
    def validate_data_integrity(self, data: pd.DataFrame, price_jump_threshold: float = 0.2,
                                volume_anomaly_threshold: float = 5) -> Dict[str, Any]:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для перевірки цілісності")
            return {"empty_data": True}

        data = self.anomaly_detector.ensure_float(data)

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
            self.anomaly_detector.validate_datetime_index(data, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці часового індексу: {str(e)}")
            issues["datetime_index_error"] = str(e)

        # Перевірка цінових даних
        try:
            self.anomaly_detector.validate_price_data(data, price_jump_threshold, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці цінових даних: {str(e)}")
            issues["price_validation_error"] = str(e)

        # Перевірка даних об'єму
        try:
            self.anomaly_detector.validate_volume_data(data, volume_anomaly_threshold, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці даних об'єму: {str(e)}")
            issues["volume_validation_error"] = str(e)

        # Перевірка на NaN і infinite значення
        try:
            self.anomaly_detector.validate_data_values(data, issues)
        except Exception as e:
            self.logger.error(f"Помилка при перевірці значень даних: {str(e)}")
            issues["data_values_error"] = str(e)

        return issues

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:

        if not isinstance(data.index, pd.DatetimeIndex):
            return data

        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            self.logger.info(f"Знайдено {dup_count} дублікатів індексу, видалення...")
            return data[~data.index.duplicated(keep='first')]

        return data
    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:

        self.logger.info(f"Підготовка DataFrame: {data.shape[0]} рядків, {data.shape[1]} стовпців")
        result = data.copy()

        # Конвертація індексу у DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                time_cols = [col for col in result.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    self.logger.info(f"Конвертування колонки {time_col} в індекс часу")
                    result[time_col] = pd.to_datetime(result[time_col], errors='coerce')
                    result.set_index(time_col, inplace=True)
                else:
                    self.logger.warning("Не знайдено колонку з часом, індекс залишається незмінним")
            except Exception as e:
                self.logger.error(f"Помилка при конвертуванні індексу: {str(e)}")

        # Сортування індексу
        result = result.sort_index()

        # Конвертація числових колонок
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in result.columns:
                if not pd.api.types.is_numeric_dtype(result[col]):
                    self.logger.info(f"Конвертування колонки {col} в числовий тип")
                    result[col] = pd.to_numeric(result[col], errors='coerce')

        return result
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'interpolate',
                              symbol: str = None, timeframe: str = None,
                              fetch_missing: bool = True) -> pd.DataFrame:
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки відсутніх значень")
            return data if data is not None else pd.DataFrame()

        integrity_issues = self.validate_data_integrity(data)
        if integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in integrity_issues.values())
            self.logger.warning(f"Перед обробкою відсутніх значень знайдено {issue_count} проблем з цілісністю даних")
            # Перевіряємо, чи є проблеми з відсутніми значеннями серед виявлених проблем
            if "columns_with_na" in integrity_issues:
                na_cols = list(integrity_issues["columns_with_na"].keys())
                self.logger.info(f"Виявлені колонки з відсутніми значеннями: {na_cols}")

        result = data.copy()
        missing_values = result.isna().sum()
        total_missing = missing_values.sum()

        if total_missing == 0:
            self.logger.info("Відсутні значення не знайдено")
            return result

        self.logger.info(
            f"Знайдено {total_missing} відсутніх значень у {len(missing_values[missing_values > 0])} колонках")

        #  Підтягування з Binance
        if isinstance(result.index, pd.DatetimeIndex) and fetch_missing and symbol and timeframe:
            time_diff = result.index.to_series().diff()
            expected_diff = time_diff.dropna().median() if len(time_diff) > 5 else None

            if expected_diff:
                missing_periods = self._detect_missing_periods(result, expected_diff)
                if missing_periods:
                    self.logger.info(f"Знайдено {len(missing_periods)} прогалин. Підтягуємо з Binance...")
                    filled = self._fetch_missing_data_from_binance(result, missing_periods, symbol, timeframe)
                    if not filled.empty:
                        result = pd.concat([result, filled])
                        result = result[~result.index.duplicated(keep='last')].sort_index()

        filled_values = 0
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        if method == 'interpolate':
            self.logger.info("Застосування методу лінійної інтерполяції")
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
            other_cols = [col for col in numeric_cols if col not in price_cols]
            before_fill = result.count().sum()

            if price_cols:
                if isinstance(result.index, pd.DatetimeIndex):
                    result[price_cols] = result[price_cols].interpolate(method='time')
                else:
                    result[price_cols] = result[price_cols].interpolate(method='linear')

            if other_cols:
                result[other_cols] = result[other_cols].interpolate().ffill().bfill()

            result = result.ffill().bfill()
            filled_values = result.count().sum() - before_fill

        elif method == 'ffill':
            self.logger.info("Застосування методу forward/backward fill")
            before_fill = result.count().sum()
            result = result.ffill().bfill()
            filled_values = result.count().sum() - before_fill

        elif method == 'mean':
            self.logger.info("Застосування методу заповнення середнім значенням")
            for col in numeric_cols:
                if col in result.columns and missing_values.get(col, 0) > 0:
                    if result[col].dropna().empty:
                        self.logger.warning(f"Колонка {col} не містить значень для обчислення середнього")
                        continue
                    col_mean = result[col].mean()
                    if pd.notna(col_mean):
                        before = result[col].isna().sum()
                        result[col] = result[col].fillna(col_mean)
                        filled_values += before - result[col].isna().sum()

        elif method == 'median':
            self.logger.info("Застосування методу заповнення медіанним значенням")
            for col in numeric_cols:
                if col in result.columns and missing_values.get(col, 0) > 0:
                    if result[col].dropna().empty:
                        self.logger.warning(f"Колонка {col} не містить значень для обчислення медіани")
                        continue
                    col_median = result[col].median()
                    if pd.notna(col_median):
                        before = result[col].isna().sum()
                        result[col] = result[col].fillna(col_median)
                        filled_values += before - result[col].isna().sum()

        else:
            self.logger.warning(f"Метод заповнення '{method}' не підтримується")

        remaining_missing = result.isna().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Залишилося {remaining_missing} незаповнених значень після обробки")

        self.logger.info(f"Заповнено {filled_values} відсутніх значень методом '{method}'")

        clean_integrity_issues = self.validate_data_integrity(result)
        if clean_integrity_issues:
            issue_count = sum(len(issues) if isinstance(issues, list) or isinstance(issues, dict) else 1
                              for issues in clean_integrity_issues.values())
            self.logger.info(f"Після обробки відсутніх значень залишилось {issue_count} проблем з цілісністю даних")

            # Перевіряємо, чи залишились проблеми з відсутніми значеннями
            if "columns_with_na" in clean_integrity_issues:
                na_cols = list(clean_integrity_issues["columns_with_na"].keys())
                self.logger.warning(f"Після обробки все ще є колонки з відсутніми значеннями: {na_cols}")
        else:
            self.logger.info("Після обробки відсутніх значень проблем з цілісністю даних не виявлено")

        return result

    def _detect_missing_periods(self, data: pd.DataFrame, expected_diff: pd.Timedelta) -> List[
        Tuple[datetime, datetime]]:

        if not isinstance(data.index, pd.DatetimeIndex) or data.empty:
            return []

        if expected_diff is None:
            self.logger.warning("expected_diff є None, неможливо визначити пропущені періоди")
            return []

        sorted_index = data.index.sort_values()

        time_diff = sorted_index.to_series().diff()
        # Використовуємо більш безпечне порівняння
        large_gaps = time_diff[time_diff > expected_diff * 1.5]

        missing_periods = []
        for timestamp, gap in large_gaps.items():
            prev_timestamp = timestamp - gap

            # Запобігаємо потенційному переповненню при обчисленні missing_steps
            try:
                missing_steps = max(0, int(gap / expected_diff) - 1)
                if missing_steps > 0:
                    self.logger.info(
                        f"Виявлено проміжок: {prev_timestamp} - {timestamp} ({missing_steps} пропущених записів)")
                    missing_periods.append((prev_timestamp, timestamp))
            except (OverflowError, ZeroDivisionError) as e:
                self.logger.error(f"Помилка при обчисленні missing_steps: {str(e)}")

        return missing_periods

    def _fetch_missing_data_from_binance(self, data: pd.DataFrame,
                                         missing_periods: List[Tuple[datetime, datetime]],
                                         symbol: str, interval: str) -> pd.DataFrame:
        if data is None or data.empty or not missing_periods:
            self.logger.warning("Отримано порожній DataFrame або немає missing_periods для заповнення даними")
            return pd.DataFrame()

        if not symbol or not interval:
            self.logger.error("Невалідний symbol або interval")
            return pd.DataFrame()

        try:
            from binance.client import Client
            api_key = BINANCE_API_KEY
            api_secret = BINANCE_API_SECRET

            if not api_key or not api_secret:
                self.logger.error("Не знайдено ключі API Binance")
                return pd.DataFrame()

            client = Client(api_key, api_secret)

            valid_intervals = ['1m', '1h', '4h', '1d']
            if interval not in valid_intervals:
                self.logger.error(f"Невалідний інтервал: {interval}")
                return pd.DataFrame()

            new_data_frames = []

            for start_time, end_time in missing_periods:
                try:
                    self.logger.info(f" Отримання даних з Binance: {symbol}, {interval}, {start_time} - {end_time}")
                    start_ms = int(start_time.timestamp() * 1000)
                    end_ms = int(end_time.timestamp() * 1000)
                    self.logger.info(f"Запит до Binance: {start_time} -> {start_ms} мс, {end_time} -> {end_ms} мс")

                    klines = client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_ms,
                        end_str=end_ms
                    )

                    if not klines:
                        self.logger.warning(f" Порожній результат з Binance: {start_time} - {end_time}")
                        continue

                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                               'close_time', 'quote_asset_volume', 'number_of_trades',
                               'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
                    binance_df = pd.DataFrame(klines, columns=columns[:min(len(columns), len(klines[0]) if klines else 0)])

                    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'], unit='ms')
                    binance_df.set_index('timestamp', inplace=True)

                    # Конвертація числових значень
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in binance_df.columns:
                            binance_df[col] = pd.to_numeric(binance_df[col], errors='coerce')

                    binance_df['is_closed'] = True

                    # Вибираємо лише ті колонки, які є в обох DataFrame
                    common_cols = data.columns.intersection(binance_df.columns)
                    if common_cols.empty:
                        self.logger.warning("⚠️ Немає спільних колонок для об'єднання")
                        continue

                    new_data = binance_df[common_cols]
                    new_data_frames.append(new_data)

                    self.logger.info(f"✅ Отримано {len(new_data)} нових записів")

                except Exception as e:
                    self.logger.error(f" Помилка при запиті Binance: {e}")

            if not new_data_frames:
                return pd.DataFrame()

            combined_new = pd.concat(new_data_frames)
            self.logger.info(f" Загалом додано {len(combined_new)} нових рядків після об'єднання")

            return combined_new

        except ImportError:
            self.logger.error(" Модуль binance не встановлено.")
            return pd.DataFrame()

    def normalize_data(self, data: pd.DataFrame, method: str = 'z-score',
                       columns: List[str] = None, exclude_columns: List[str] = None) -> Tuple[
        pd.DataFrame, Optional[Dict]]:
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для нормалізації")
            return (data if data is not None else pd.DataFrame()), None

        result = data.copy()

        # Якщо не вказано колонки, беремо лише основні OHLCV
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']

        # Фільтруємо тільки ті колонки, які існують у DataFrame
        normalize_cols = [col for col in columns if col in result.columns]

        if not normalize_cols:
            self.logger.warning("Немає числових колонок для нормалізації")
            return result, None

        try:
            # Зберігаємо оригінальні дані для відновлення у разі помилки
            original_values = result[normalize_cols].copy()

            # Перевіряємо наявність цінових колонок (OHLC)
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in normalize_cols]
            other_cols = [col for col in normalize_cols if col not in price_cols]

            # Використовуємо SimpleImputer для заповнення NaN значень
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')

            scaler_meta = {
                'method': method,
                'columns': normalize_cols,
                'imputers': {},
                'scalers': {},
                'original_mins': {},  # Зберігаємо мінімальні значення для відновлення
                'scaling_factors': {}  # Зберігаємо фактори масштабування
            }

            if price_cols:
                # Спочатку обробляємо цінові колонки, зберігаючи їх співвідношення

                # 1. Зберігаємо оригінальні співвідношення між цінами
                if all(col in price_cols for col in ['high', 'low']):
                    result['_hl_ratio'] = result['high'] / result['low']  # High-Low ratio
                    # Замінюємо нескінченні значення та NaN на 1.01 (невелике перебільшення)
                    result['_hl_ratio'].replace([float('inf'), float('-inf')], 1.01, inplace=True)
                    result['_hl_ratio'].fillna(1.01, inplace=True)

                if all(col in price_cols for col in ['high', 'open']):
                    result['_ho_ratio'] = result['high'] / result['open']
                    result['_ho_ratio'].replace([float('inf'), float('-inf')], 1.01, inplace=True)
                    result['_ho_ratio'].fillna(1.01, inplace=True)

                if all(col in price_cols for col in ['close', 'low']):
                    result['_cl_ratio'] = result['close'] / result['low']
                    result['_cl_ratio'].replace([float('inf'), float('-inf')], 1.01, inplace=True)
                    result['_cl_ratio'].fillna(1.01, inplace=True)

                if all(col in price_cols for col in ['open', 'close']):
                    result['_oc_ratio'] = result['open'] / result['close']
                    result['_oc_ratio'].replace([float('inf'), float('-inf')], 1.01, inplace=True)
                    result['_oc_ratio'].fillna(1.01, inplace=True)

                # 2. Нормалізуємо базову ціну (наприклад, low)
                if 'low' in price_cols:
                    base_col = 'low'
                elif 'close' in price_cols:
                    base_col = 'close'
                elif 'open' in price_cols:
                    base_col = 'open'
                else:
                    base_col = 'high'

                # Імпутація пропущених значень для базової колонки
                base_imputer = SimpleImputer(strategy='mean')
                base_values = base_imputer.fit_transform(result[[base_col]])

                # Зберігаємо мінімальне значення для забезпечення невід'ємності
                original_min = float(base_values.min())
                scaler_meta['original_mins'][base_col] = original_min

                # Вибір скейлера для базової колонки з гарантуванням невід'ємних значень
                if method == 'z-score':
                    from sklearn.preprocessing import StandardScaler
                    # Зсув даних, щоб мінімум був > 0 перед масштабуванням
                    shift_value = abs(min(0, original_min)) + 0.01  # Додаємо 0.01 для гарантії додатності
                    base_values = base_values + shift_value

                    # Зберігаємо зсув для подальшого використання
                    scaler_meta['scaling_factors'][f'{base_col}_shift'] = shift_value

                    base_scaler = StandardScaler()
                    base_scaled = base_scaler.fit_transform(base_values)

                    # Після z-score масштабування переводимо значення у додатній діапазон
                    min_scaled = float(base_scaled.min())
                    shift_after = abs(min(0, min_scaled)) + 0.01
                    base_scaled = base_scaled + shift_after
                    scaler_meta['scaling_factors'][f'{base_col}_after_shift'] = shift_after

                elif method == 'min-max':
                    from sklearn.preprocessing import MinMaxScaler
                    # Використовуємо MinMaxScaler з додатнім діапазоном
                    base_scaler = MinMaxScaler(feature_range=(0.01, 1.0))
                    base_scaled = base_scaler.fit_transform(base_values)

                elif method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    # Для RobustScaler також потрібно зсунути дані
                    shift_value = abs(min(0, original_min)) + 0.01
                    base_values = base_values + shift_value
                    scaler_meta['scaling_factors'][f'{base_col}_shift'] = shift_value

                    base_scaler = RobustScaler()
                    base_scaled = base_scaler.fit_transform(base_values)

                    # Гарантуємо додатність після масштабування
                    min_scaled = float(base_scaled.min())
                    shift_after = abs(min(0, min_scaled)) + 0.01
                    base_scaled = base_scaled + shift_after
                    scaler_meta['scaling_factors'][f'{base_col}_after_shift'] = shift_after
                else:
                    self.logger.error(f"Непідтримуваний метод нормалізації: {method}")
                    return result, None

                # Оновлюємо базову колонку нормалізованими значеннями
                result[base_col] = base_scaled

                # Зберігаємо скейлер і імпутер для базової колонки
                scaler_meta['imputers'][base_col] = base_imputer
                scaler_meta['scalers'][base_col] = base_scaler

                # 3. Відновлюємо інші цінові колонки на основі збережених співвідношень
                if 'high' in price_cols and base_col != 'high':
                    if '_hl_ratio' in result.columns and base_col == 'low':
                        # Відновлюємо high на основі нормалізованого low та оригінального співвідношення
                        # Гарантуємо, що співвідношення high/low >= 1
                        result['_hl_ratio'] = result['_hl_ratio'].clip(lower=1.001)
                        result['high'] = result['low'] * result['_hl_ratio']
                    else:
                        # Окрема нормалізація для high з гарантією невід'ємності
                        result['high'] = self._normalize_single_column_non_negative(result, 'high', method, scaler_meta)

                if 'open' in price_cols and base_col != 'open':
                    if base_col == 'low' and '_ho_ratio' in result.columns:
                        # Відновлюємо open відносно low, використовуючи співвідношення
                        result['open'] = result['low'] * result['_ho_ratio']
                    else:
                        result['open'] = self._normalize_single_column_non_negative(result, 'open', method, scaler_meta)

                if 'close' in price_cols and base_col != 'close':
                    if base_col == 'low' and '_cl_ratio' in result.columns:
                        # Гарантуємо, що співвідношення close/low >= 1
                        result['_cl_ratio'] = result['_cl_ratio'].clip(lower=1.0)
                        result['close'] = result['low'] * result['_cl_ratio']
                    else:
                        result['close'] = self._normalize_single_column_non_negative(result, 'close', method,
                                                                                     scaler_meta)

            # Обробляємо інші колонки окремо (наприклад, volume) - завжди невід'ємні
            for col in other_cols:
                result[col] = self._normalize_single_column_non_negative(result, col, method, scaler_meta)

            # Видаляємо тимчасові колонки співвідношень
            for col in result.columns:
                if col.startswith('_') and col.endswith('_ratio'):
                    result.drop(col, axis=1, inplace=True)

            # Перевірка результатів
            if 'high' in result.columns and 'low' in result.columns:
                invalid_rows = (result['high'] < result['low']).sum()
                if invalid_rows > 0:
                    self.logger.warning(f"Після нормалізації виявлено {invalid_rows} рядків з high < low")
                    # Виправляємо проблему, встановлюючи high = low * 1.001
                    mask = result['high'] < result['low']
                    result.loc[mask, 'high'] = result.loc[mask, 'low'] * 1.001

            # Перевірка наявності від'ємних значень
            for col in normalize_cols:
                neg_count = (result[col] < 0).sum()
                if neg_count > 0:
                    self.logger.warning(f"Після нормалізації виявлено {neg_count} від'ємних значень у колонці {col}")
                    # Виправляємо проблему, зсуваючи всі значення для забезпечення невід'ємності
                    min_val = result[col].min()
                    shift = abs(min_val) + 0.01
                    result[col] = result[col] + shift
                    if col not in scaler_meta['scaling_factors']:
                        scaler_meta['scaling_factors'][f'{col}_final_shift'] = shift

            return result, scaler_meta

        except Exception as e:
            self.logger.error(f"Помилка при нормалізації даних: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Повертаємо оригінальні дані у разі помилки
            return data.copy(), None

    def _normalize_single_column_non_negative(self, df: pd.DataFrame, column: str, method: str,
                                              scaler_meta: Dict) -> pd.Series:
        """Нормалізує окрему колонку з використанням вказаного методу, гарантуючи невід'ємні значення."""
        from sklearn.impute import SimpleImputer

        # Імпутація пропущених значень
        imputer = SimpleImputer(strategy='mean')
        values = imputer.fit_transform(df[[column]])

        # Зберігаємо оригінальний мінімум для відстеження
        original_min = float(values.min())
        scaler_meta['original_mins'][column] = original_min

        # Вибір скейлера з гарантією невід'ємних значень
        if method == 'z-score':
            # Зсуваємо дані, щоб мінімум був додатній
            shift_value = abs(min(0, original_min)) + 0.01
            values = values + shift_value
            scaler_meta['scaling_factors'][f'{column}_shift'] = shift_value

            scaler = StandardScaler()
            scaled = scaler.fit_transform(values)

            # Після z-score масштабування переводимо значення у додатній діапазон
            min_scaled = float(scaled.min())
            shift_after = abs(min(0, min_scaled)) + 0.01
            scaled = scaled + shift_after
            scaler_meta['scaling_factors'][f'{column}_after_shift'] = shift_after

        elif method == 'min-max':
            from sklearn.preprocessing import MinMaxScaler
            # Використовуємо MinMaxScaler з додатнім діапазоном
            scaler = MinMaxScaler(feature_range=(0.01, 1.0))
            scaled = scaler.fit_transform(values)

        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            # Для RobustScaler також потрібно зсунути дані
            shift_value = abs(min(0, original_min)) + 0.01
            values = values + shift_value
            scaler_meta['scaling_factors'][f'{column}_shift'] = shift_value

            scaler = RobustScaler()
            scaled = scaler.fit_transform(values)

            # Гарантуємо додатність після масштабування
            min_scaled = float(scaled.min())
            shift_after = abs(min(0, min_scaled)) + 0.01
            scaled = scaled + shift_after
            scaler_meta['scaling_factors'][f'{column}_after_shift'] = shift_after
        else:
            raise ValueError(f"Непідтримуваний метод нормалізації: {method}")

        # Фінальна перевірка на від'ємні значення
        if (scaled < 0).any():
            # Зсуваємо все до додатніх значень
            min_val = float(scaled.min())
            final_shift = abs(min_val) + 0.01
            scaled = scaled + final_shift
            scaler_meta['scaling_factors'][f'{column}_final_shift'] = final_shift

        # Зберігаємо метадані
        scaler_meta['imputers'][column] = imputer
        scaler_meta['scalers'][column] = scaler

        return pd.Series(scaled.flatten(), index=df.index)

    def add_time_features_safely(self, data: pd.DataFrame, cyclical: bool = True,
                                 add_sessions: bool = False, tz: str = 'Europe/Kiev') -> pd.DataFrame:
        """Додає часові ознаки безпечно, без перезапису існуючих колонок"""
        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для додавання часових ознак")
            return data if data is not None else pd.DataFrame()

        # Зберігаємо копію вхідних даних
        result = data.copy()

        # Зберігаємо список оригінальних колонок
        original_columns = result.columns.tolist()

        # Перевірка та конвертація індексу в DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in result.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    result[time_col] = pd.to_datetime(result[time_col], errors='coerce')
                    result.set_index(time_col, inplace=True)

                    # Перевірка на NaT після конвертації
                    if result.index.isna().any():
                        nat_count = result.index.isna().sum()
                        self.logger.warning(f"Знайдено {nat_count} NaT значень в індексі після конвертації")
                        result = result[~result.index.isna()]
                else:
                    self.logger.error("Неможливо додати часові ознаки: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        self.logger.info("Додавання часових ознак")

        # Обробка часових поясів з обробкою помилок
        try:
            if result.index.tz is None:
                self.logger.info(f"Встановлення часового поясу {tz}")
                try:
                    # Спроба звичайної локалізації
                    result.index = result.index.tz_localize(tz)
                except pytz.exceptions.NonExistentTimeError:
                    # Перехід на літній час: пропущена година
                    self.logger.warning(
                        "Виявлено час, що не існує при переході на літній час. Використання нестрогої локалізації.")
                    try:
                        # Зсув вперед
                        result.index = result.index.tz_localize(tz, nonexistent='shift_forward')
                    except Exception:
                        try:
                            # Зсув назад як альтернатива
                            result.index = result.index.tz_localize(tz, nonexistent='shift_backward')
                        except Exception:
                            # Створення NaT як остання спроба
                            self.logger.warning("Використання методу 'NaT' для неіснуючих значень часу")
                            result.index = result.index.tz_localize(tz, nonexistent='NaT')
                            # Видалення NaT
                            result = result[~result.index.isna()]
                except pytz.exceptions.AmbiguousTimeError:
                    # Перехід на зимовий час: дубльована година
                    self.logger.warning(
                        "Виявлено неоднозначний час при переході на зимовий час. Використання стратегії.")
                    try:
                        # Спочатку пробуємо False
                        result.index = result.index.tz_localize(tz, ambiguous=False)
                    except Exception:
                        try:
                            # Якщо не вдалося, спробуємо True
                            result.index = result.index.tz_localize(tz, ambiguous=True)
                        except Exception:
                            # Як остання спроба, використовуємо індивідуальний підхід
                            temp_index = pd.DatetimeIndex([
                                timestamp.tz_localize(tz, ambiguous=False)
                                if pd.Timestamp(timestamp).fold == 0 else timestamp.tz_localize(tz, ambiguous=True)
                                for timestamp in result.index
                            ])
                            result.index = temp_index
            elif result.index.tz.zone != tz:
                self.logger.info(f"Конвертація часового поясу з {result.index.tz.zone} в {tz}")
                try:
                    result.index = result.index.tz_convert(tz)
                except Exception as e:
                    self.logger.warning(
                        f"Помилка при конвертації часового поясу: {str(e)}. Продовжуємо з поточним часовим поясом.")
        except Exception as e:
            self.logger.error(f"Загальна помилка при обробці часового поясу: {str(e)}")

        # Створюємо імена для нових колонок
        time_feature_names = []

        # Базові часові ознаки
        time_features = {}
        time_features['tf_hour'] = result.index.hour
        time_features['tf_day'] = result.index.day
        time_features['tf_weekday'] = result.index.weekday

        # Безпечне отримання номера тижня
        try:
            if hasattr(result.index, 'isocalendar') and callable(result.index.isocalendar):
                isocal = result.index.isocalendar()
                if isinstance(isocal, pd.DataFrame):  # pandas >= 1.1.0
                    time_features['tf_week'] = isocal['week']
                else:  # старіші версії pandas
                    time_features['tf_week'] = [x[1] for x in isocal]
            else:
                # Альтернативний метод
                time_features['tf_week'] = result.index.to_series().apply(lambda x: x.isocalendar()[1])
        except Exception as e:
            self.logger.warning(f"Помилка при отриманні номера тижня: {str(e)}. Використовуємо альтернативний метод.")
            # Запасний варіант
            time_features['tf_week'] = result.index.week if hasattr(result.index,
                                                                    'week') else result.index.to_series().dt.week

        time_features['tf_month'] = result.index.month
        time_features['tf_quarter'] = result.index.quarter
        time_features['tf_year'] = result.index.year
        time_features['tf_dayofyear'] = result.index.dayofyear

        # Бінарні ознаки
        time_features['tf_is_weekend'] = (time_features['tf_weekday'].isin([5, 6])).astype(int)
        time_features['tf_is_month_start'] = result.index.is_month_start.astype(int)
        time_features['tf_is_month_end'] = result.index.is_month_end.astype(int)
        time_features['tf_is_quarter_start'] = result.index.is_quarter_start.astype(int)
        time_features['tf_is_quarter_end'] = result.index.is_quarter_end.astype(int)
        time_features['tf_is_year_start'] = result.index.is_year_start.astype(int)
        time_features['tf_is_year_end'] = result.index.is_year_end.astype(int)

        # Додаємо префікс tf_ до всіх часових ознак для уникнення конфліктів
        time_feature_names.extend(time_features.keys())

        # Циклічні ознаки
        if cyclical:
            self.logger.info("Додавання циклічних ознак")

            time_features['tf_hour_sin'] = np.sin(2 * np.pi * time_features['tf_hour'] / 24)
            time_features['tf_hour_cos'] = np.cos(2 * np.pi * time_features['tf_hour'] / 24)

            # Безпечне обчислення кількості днів у місяці
            try:
                days_in_month = result.index.days_in_month
            except AttributeError:
                self.logger.warning("Атрибут days_in_month не знайдений. Використовуємо стандартне значення 30.")
                days_in_month = pd.Series([30] * len(result), index=result.index)

            # Перевірка на нульові значення у знаменнику
            days_in_month = pd.Series(days_in_month).replace(0, 30).values

            time_features['tf_day_sin'] = np.sin(2 * np.pi * time_features['tf_day'] / days_in_month)
            time_features['tf_day_cos'] = np.cos(2 * np.pi * time_features['tf_day'] / days_in_month)

            time_features['tf_weekday_sin'] = np.sin(2 * np.pi * time_features['tf_weekday'] / 7)
            time_features['tf_weekday_cos'] = np.cos(2 * np.pi * time_features['tf_weekday'] / 7)

            time_features['tf_week_sin'] = np.sin(2 * np.pi * time_features['tf_week'] / 52)
            time_features['tf_week_cos'] = np.cos(2 * np.pi * time_features['tf_week'] / 52)

            time_features['tf_month_sin'] = np.sin(2 * np.pi * time_features['tf_month'] / 12)
            time_features['tf_month_cos'] = np.cos(2 * np.pi * time_features['tf_month'] / 12)

            time_features['tf_quarter_sin'] = np.sin(2 * np.pi * time_features['tf_quarter'] / 4)
            time_features['tf_quarter_cos'] = np.cos(2 * np.pi * time_features['tf_quarter'] / 4)

            # Додаємо назви циклічних ознак
            time_feature_names.extend([
                'tf_hour_sin', 'tf_hour_cos', 'tf_day_sin', 'tf_day_cos',
                'tf_weekday_sin', 'tf_weekday_cos', 'tf_week_sin', 'tf_week_cos',
                'tf_month_sin', 'tf_month_cos', 'tf_quarter_sin', 'tf_quarter_cos'
            ])

        # Торгові сесії
        if add_sessions:
            self.logger.info("Додавання індикаторів торгових сесій")

            # Азійська сесія: 00:00-09:00
            time_features['tf_asian_session'] = ((time_features['tf_hour'] >= 0) &
                                                 (time_features['tf_hour'] < 9)).astype(int)

            # Європейська сесія: 08:00-17:00
            time_features['tf_european_session'] = ((time_features['tf_hour'] >= 8) &
                                                    (time_features['tf_hour'] < 17)).astype(int)

            # Американська сесія: 13:00-22:00
            time_features['tf_american_session'] = ((time_features['tf_hour'] >= 13) &
                                                    (time_features['tf_hour'] < 22)).astype(int)

            # Перекриття сесій
            time_features['tf_asia_europe_overlap'] = ((time_features['tf_hour'] >= 8) &
                                                       (time_features['tf_hour'] < 9)).astype(int)
            time_features['tf_europe_america_overlap'] = ((time_features['tf_hour'] >= 13) &
                                                          (time_features['tf_hour'] < 17)).astype(int)

            # Неактивні години (22:00-00:00)
            time_features['tf_inactive_hours'] = (time_features['tf_hour'] >= 22).astype(int)

            # Додаємо назви ознак сесій
            time_feature_names.extend([
                'tf_asian_session', 'tf_european_session', 'tf_american_session',
                'tf_asia_europe_overlap', 'tf_europe_america_overlap', 'tf_inactive_hours'
            ])

        # Перевіряємо наявність конфліктів із існуючими колонками
        conflicts = [feature for feature in time_feature_names if feature in original_columns]
        if conflicts:
            # Замість перезапису додаємо унікальний суфікс до конфліктуючих колонок
            self.logger.warning(f"Виявлено потенційні конфлікти імен колонок: {conflicts}")

            for feature_name in conflicts:
                i = 1
                new_name = f"{feature_name}_{i}"
                while new_name in original_columns:
                    i += 1
                    new_name = f"{feature_name}_{i}"

                self.logger.info(f"Перейменування конфліктної колонки {feature_name} на {new_name}")

                # Оновлюємо ім'я в словнику та списку
                if feature_name in time_features:
                    time_features[new_name] = time_features.pop(feature_name)
                    time_feature_names.remove(feature_name)
                    time_feature_names.append(new_name)

        # Додаємо часові ознаки до DataFrame
        for feature_name, feature_values in time_features.items():
            result[feature_name] = feature_values

        # Перевіряємо, чи всі оригінальні колонки збереглися
        for col in original_columns:
            if col not in result.columns:
                self.logger.error(f"Колонка {col} зникла після додавання часових ознак. Відновлення...")
                if col in data.columns:
                    result[col] = data[col]

        self.logger.info(f"Успішно додано {len(time_features)} часових ознак")
        return result



    def remove_duplicate_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для обробки дублікатів")
            return data if data is not None else pd.DataFrame()

        original_shape = data.shape

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')

                    # Перевірка на NaT
                    if data[time_col].isna().any():
                        self.logger.warning(f"Знайдено NaT значення в колонці {time_col}")
                        data = data[~data[time_col].isna()]

                    data.set_index(time_col, inplace=True)
                else:
                    self.logger.error("Неможливо виявити дублікати: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        # Перевірка на наявність дублікатів
        duplicates = data.index.duplicated()
        duplicates_count = duplicates.sum()

        if duplicates_count == 0:
            self.logger.info("Дублікати часових міток не знайдено")
            return data

        self.logger.info(f"Знайдено {duplicates_count} дублікатів часових міток")

        # Зберігаємо унікальні індекси, видаляємо дублікати і сортуємо
        result = data[~duplicates].sort_index()

        # Логування результатів
        self.logger.info(
            f"Видалено {duplicates_count} дублікатів. Вхідний розмір {original_shape}, залишилось {result.shape} записів.")

        return result

    def filter_by_time_range(self, data: pd.DataFrame,
                             start_time: Optional[Union[str, datetime]] = None,
                             end_time: Optional[Union[str, datetime]] = None) -> pd.DataFrame:

        if data is None or data.empty:
            self.logger.warning("Отримано порожній DataFrame для фільтрації")
            return data if data is not None else pd.DataFrame()

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Індекс не є DatetimeIndex. Спроба конвертувати.")
            try:
                time_cols = [col for col in data.columns if
                             any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
                if time_cols:
                    time_col = time_cols[0]
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')

                    # Перевірка на NaT
                    if data[time_col].isna().any():
                        nat_count = data[time_col].isna().sum()
                        self.logger.warning(f"Знайдено {nat_count} NaT значень в колонці {time_col}")
                        data = data[~data[time_col].isna()]

                    data.set_index(time_col, inplace=True)
                else:
                    self.logger.error("Неможливо фільтрувати за часом: не знайдено часову колонку")
                    return data
            except Exception as e:
                self.logger.error(f"Помилка при конвертації індексу: {str(e)}")
                return data

        result = data.copy()
        initial_count = len(result)

        start_dt = None
        end_dt = None

        # Обробка початкового часу
        if start_time is not None:
            try:
                start_dt = pd.to_datetime(start_time)

                # Коректна обробка часових поясів
                if result.index.tz is not None and start_dt.tz is None:
                    self.logger.info(f"Додавання часового поясу {result.index.tz} до початкового часу")
                    start_dt = start_dt.tz_localize(result.index.tz)
                elif result.index.tz is None and start_dt.tz is not None:
                    self.logger.info(f"Видалення часового поясу з початкового часу")
                    start_dt = start_dt.tz_localize(None)
                elif result.index.tz is not None and start_dt.tz is not None and result.index.tz != start_dt.tz:
                    self.logger.info(f"Конвертація часового поясу з {start_dt.tz} в {result.index.tz}")
                    start_dt = start_dt.tz_convert(result.index.tz)

                result = result[result.index >= start_dt]
                self.logger.info(f"Фільтрація за початковим часом: {start_dt}")
            except Exception as e:
                self.logger.error(f"Помилка при конвертації початкового часу: {str(e)}")

        # Обробка кінцевого часу
        if end_time is not None:
            try:
                end_dt = pd.to_datetime(end_time)

                # Коректна обробка часових поясів
                if result.index.tz is not None and end_dt.tz is None:
                    self.logger.info(f"Додавання часового поясу {result.index.tz} до кінцевого часу")
                    end_dt = end_dt.tz_localize(result.index.tz)
                elif result.index.tz is None and end_dt.tz is not None:
                    self.logger.info(f"Видалення часового поясу з кінцевого часу")
                    end_dt = end_dt.tz_localize(None)
                elif result.index.tz is not None and end_dt.tz is not None and result.index.tz != end_dt.tz:
                    self.logger.info(f"Конвертація часового поясу з {end_dt.tz} в {result.index.tz}")
                    end_dt = end_dt.tz_convert(result.index.tz)

                result = result[result.index <= end_dt]
                self.logger.info(f"Фільтрація за кінцевим часом: {end_dt}")
            except Exception as e:
                self.logger.error(f"Помилка при конвертації кінцевого часу: {str(e)}")

        final_count = len(result)

        # Перевірка логічної відповідності початкової та кінцевої дати
        if start_dt is not None and end_dt is not None:
            if start_dt > end_dt:
                self.logger.warning(
                    f"Початковий час ({start_dt}) пізніше кінцевого часу ({end_dt}). Результат може бути порожнім.")
                if len(result) == 0:
                    self.logger.warning("Після фільтрації отримано порожній DataFrame")

        self.logger.info(f"Відфільтровано {initial_count - final_count} записів. Залишилось {final_count} записів.")
        return result

