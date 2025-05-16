import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging
from data.db import DatabaseManager
from utils.config import db_connection

class FeatureEngineering:

    def __init__(self, log_level=logging.INFO):
        self.log_level = log_level
        self.db_connection = db_connection
        self.db_manager = DatabaseManager()
        self.supported_symbols = self.db_manager.supported_symbols
        logging.basicConfig(level=self.log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ініціалізація класу...")
        self.ready = True

    def create_target_variable(self, data: pd.DataFrame,
                               price_column: str = 'close',
                               horizon: int = 1,
                               target_type: str = 'return') -> pd.DataFrame:

        self.logger.info(f"Створення цільової змінної типу '{target_type}' з горизонтом {horizon}")

        # Створюємо копію, щоб не модифікувати оригінальні дані
        result_df = data.copy()

        # Перевіряємо, що price_column існує в даних
        if price_column not in result_df.columns:
            self.logger.error(f"Стовпець {price_column} не знайдено в даних")
            raise ValueError(f"Стовпець {price_column} не знайдено в даних")

        # Перевіряємо, що індекс часовий для правильного зсуву
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("Індекс даних не є часовим (DatetimeIndex). Цільова змінна може бути неточною.")

        # Перевіряємо наявність пропущених значень у стовпці ціни
        if result_df[price_column].isna().any():
            self.logger.warning(f"Стовпець {price_column} містить NaN значення, вони будуть заповнені")
            result_df[price_column] = result_df[price_column].fillna(method='ffill').fillna(method='bfill')

        # Створюємо цільову змінну в залежності від типу
        if target_type == 'return':
            # Процентна зміна ціни через horizon періодів
            target_name = f'target_return_{horizon}p'
            result_df[target_name] = result_df[price_column].pct_change(periods=-horizon).shift(horizon)
            self.logger.info(f"Створено цільову змінну '{target_name}' як процентну зміну ціни")

        elif target_type == 'log_return':
            # Логарифмічна зміна ціни
            target_name = f'target_log_return_{horizon}p'
            result_df[target_name] = np.log(result_df[price_column].shift(-horizon) / result_df[price_column])
            self.logger.info(f"Створено цільову змінну '{target_name}' як логарифмічну зміну ціни")

        elif target_type == 'direction':
            # Напрямок зміни ціни (1 - ріст, 0 - падіння)
            target_name = f'target_direction_{horizon}p'
            future_price = result_df[price_column].shift(-horizon)
            result_df[target_name] = np.where(future_price > result_df[price_column], 1, 0)
            self.logger.info(f"Створено цільову змінну '{target_name}' як напрямок зміни ціни")

        elif target_type == 'volatility':
            # Майбутня волатильність як стандартне відхилення прибутковості за період
            target_name = f'target_volatility_{horizon}p'
            # Розраховуємо логарифмічну прибутковість
            log_returns = np.log(result_df[price_column] / result_df[price_column].shift(1))
            # Розраховуємо волатильність за наступні horizon періодів
            result_df[target_name] = log_returns.rolling(window=horizon).std().shift(-horizon)
            self.logger.info(f"Створено цільову змінну '{target_name}' як майбутню волатильність")

        elif target_type == 'price':
            # Майбутня ціна
            target_name = f'target_price_{horizon}p'
            result_df[target_name] = result_df[price_column].shift(-horizon)
            self.logger.info(f"Створено цільову змінну '{target_name}' як майбутню ціну")

        elif target_type == 'range':
            # Діапазон зміни ціни (high-low) за наступні horizon періодів
            target_name = f'target_range_{horizon}p'
            # Для точного розрахунку діапазону потрібні high і low колонки
            if 'high' in result_df.columns and 'low' in result_df.columns:
                # Знаходимо максимальне high і мінімальне low за наступні horizon періодів
                high_values = result_df['high'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
                low_values = result_df['low'].rolling(window=horizon, min_periods=1).min().shift(-horizon)
                result_df[target_name] = (high_values - low_values) / result_df[price_column]
                self.logger.info(f"Створено цільову змінну '{target_name}' як відносний діапазон ціни")
            else:
                self.logger.warning(
                    "Колонки 'high' або 'low' відсутні, використовуємо близьку ціну для розрахунку діапазону")
                # Використовуємо амплітуду зміни ціни close
                price_max = result_df[price_column].rolling(window=horizon, min_periods=1).max().shift(-horizon)
                price_min = result_df[price_column].rolling(window=horizon, min_periods=1).min().shift(-horizon)
                result_df[target_name] = (price_max - price_min) / result_df[price_column]
                self.logger.info(f"Створено цільову змінну '{target_name}' як відносний діапазон ціни")
        else:
            self.logger.error(f"Невідомий тип цільової змінної: {target_type}")
            raise ValueError(
                f"Невідомий тип цільової змінної: {target_type}. Допустимі значення: 'return', 'log_return', 'direction', 'volatility', 'price', 'range'")

        # Заповнюємо NaN значення в цільовій змінній
        if result_df[target_name].isna().any():
            self.logger.warning(
                f"Цільова змінна {target_name} містить {result_df[target_name].isna().sum()} NaN значень")
            # Для цільових змінних краще видалити рядки з NaN, ніж заповнювати їх
            if target_type in ['return', 'log_return', 'price', 'range', 'volatility']:
                # Для числових цільових змінних можна спробувати заповнити медіаною
                # Але це не рекомендується для навчання моделей
                median_val = result_df[target_name].median()
                result_df[target_name] = result_df[target_name].fillna(median_val)
                self.logger.warning(f"NaN значення в цільовій змінній заповнені медіаною: {median_val}")
            elif target_type == 'direction':
                # Для бінарної класифікації можна заповнити найбільш поширеним класом
                mode_val = result_df[target_name].mode()[0]
                result_df[target_name] = result_df[target_name].fillna(mode_val)
                self.logger.warning(f"NaN значення в цільовій змінній заповнені модою: {mode_val}")

        return result_df

    def prepare_features_pipeline(self, data: pd.DataFrame,
                                  target_column: str = 'close',
                                  horizon: int = 1,
                                  feature_groups: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:

        self.logger.info("Запуск конвеєра підготовки ознак...")

        # Перевіряємо, що цільовий стовпець існує в даних
        if target_column not in data.columns:
            self.logger.error(f"Цільовий стовпець {target_column} не знайдено в даних")
            raise ValueError(f"Цільовий стовпець {target_column} не знайдено в даних")

        # Перевіряємо, що дані мають часовий індекс
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Дані не мають часового індексу (DatetimeIndex). Це може вплинути на якість ознак.")

        # Створюємо копію даних
        result_df = data.copy()

        # Визначаємо стандартні групи ознак, якщо не вказано
        standard_groups = [
            'lagged', 'rolling', 'ewm', 'returns', 'technical', 'volatility',
            'ratio', 'crossover', 'datetime'
        ]

        if feature_groups is None:
            feature_groups = standard_groups
            self.logger.info(f"Використовуються всі стандартні групи ознак: {feature_groups}")
        else:
            # Перевіряємо валідність зазначених груп
            invalid_groups = [group for group in feature_groups if group not in standard_groups]
            if invalid_groups:
                self.logger.warning(f"Невідомі групи ознак {invalid_groups} будуть пропущені")
                feature_groups = [group for group in feature_groups if group in standard_groups]

        # Лічильник доданих ознак
        total_features_added = 0
        initial_feature_count = len(result_df.columns)

        # Створюємо ознаки для кожної групи
        for group in feature_groups:
            try:
                self.logger.info(f"Обробка групи ознак: {group}")

                if group == 'lagged':
                    # Лагові ознаки
                    price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result_df.columns]
                    result_df = self.create_lagged_features(result_df, columns=price_cols,
                                                            lag_periods=[1, 3, 5, 7, 14, 30])

                elif group == 'rolling':
                    # Ознаки на основі ковзного вікна
                    price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result_df.columns]
                    result_df = self.create_rolling_features(result_df, columns=price_cols,
                                                             window_sizes=[5, 10, 20, 50])

                elif group == 'ewm':
                    # Ознаки на основі експоненціально зваженого вікна
                    price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result_df.columns]
                    result_df = self.create_ewm_features(result_df, columns=price_cols, spans=[5, 10, 20, 50])

                elif group == 'returns':
                    # Ознаки прибутковості
                    result_df = self.create_return_features(result_df, price_column=target_column,
                                                            periods=[1, 3, 5, 7, 14, 30])

                elif group == 'technical':
                    # Технічні індикатори
                    result_df = self.create_technical_features(result_df)

                elif group == 'volatility':
                    # Ознаки волатильності
                    result_df = self.create_volatility_features(result_df, price_column=target_column,
                                                                window_sizes=[5, 10, 20, 50])

                elif group == 'ratio':
                    # Ознаки-співвідношення
                    price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result_df.columns]
                    if 'volume' in result_df.columns:
                        price_cols.append('volume')
                    result_df = self.create_ratio_features(result_df, numerators=price_cols, denominators=price_cols)

                elif group == 'crossover':
                    # Перевіряємо наявність реалізації методу
                    if hasattr(self, 'create_crossover_features'):
                        # Потрібні технічні індикатори, переважно SMA та EMA
                        # Перевіряємо, чи були вже створені
                        sma_cols = [col for col in result_df.columns if col.startswith('sma_')]
                        ema_cols = [col for col in result_df.columns if col.startswith('ema_')]

                        if not sma_cols or not ema_cols:
                            # Якщо технічні індикатори ще не створені, створюємо їх
                            if 'technical' not in feature_groups:
                                result_df = self.create_technical_features(result_df, indicators=['sma', 'ema'])

                        # Після створення перевіряємо знову наявність індикаторів
                        sma_cols = [col for col in result_df.columns if col.startswith('sma_')]
                        ema_cols = [col for col in result_df.columns if col.startswith('ema_')]

                        if sma_cols and ema_cols:
                            result_df = self.create_crossover_features(result_df, fast_columns=ema_cols,
                                                                       slow_columns=sma_cols)
                        else:
                            self.logger.warning("Не знайдено SMA або EMA індикаторів для створення ознак перетинів")
                    else:
                        self.logger.warning("Метод create_crossover_features не реалізований, пропускаємо")

                elif group == 'datetime':
                    # Ознаки дати і часу
                    if hasattr(self, 'create_datetime_features'):
                        result_df = self.create_datetime_features(result_df, cyclical=True)
                    else:
                        self.logger.warning("Метод create_datetime_features не реалізований, пропускаємо")

            except Exception as e:
                self.logger.error(f"Помилка при створенні групи ознак {group}: {str(e)}")

        # Рахуємо кількість доданих ознак
        features_added = len(result_df.columns) - initial_feature_count
        self.logger.info(f"Загалом додано {features_added} ознак")

        # Створюємо цільову змінну
        self.logger.info(f"Створення цільової змінної з горизонтом {horizon}...")
        target = result_df[target_column].shift(-horizon)

        # Видаляємо рядки з NaN у цільовій змінній (зазвичай останні рядки)
        valid_idx = ~target.isna()
        if not valid_idx.all():
            self.logger.info(f"Видалено {sum(~valid_idx)} рядків з NaN у цільовій змінній")
            result_df = result_df.loc[valid_idx]
            target = target.loc[valid_idx]

        # Перевіряємо наявність NaN у ознаках і заповнюємо їх
        nan_cols = result_df.columns[result_df.isna().any()].tolist()
        if nan_cols:
            self.logger.warning(f"Виявлено {len(nan_cols)} стовпців з NaN значеннями. Заповнюємо їх.")

            for col in nan_cols:
                # Використовуємо різні стратегії заповнення залежно від типу ознаки
                if result_df[col].dtype == 'object':
                    # Для категоріальних ознак використовуємо найчастіше значення
                    result_df[col] = result_df[col].fillna(
                        result_df[col].mode()[0] if not result_df[col].mode().empty else "unknown")
                else:
                    # Для числових ознак використовуємо медіану
                    result_df[col] = result_df[col].fillna(result_df[col].median())

        # Опціонально можна додати відбір ознак, якщо їх надто багато
        if len(result_df.columns) > 100:  # Порогове значення кількості ознак
            self.logger.info(
                f"Кількість ознак ({len(result_df.columns)}) перевищує поріг. Розгляньте використання select_features для зменшення розмірності.")

        return result_df, target

