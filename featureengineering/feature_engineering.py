import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Callable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import logging
import ta
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


def main(telegram_mode=False, bot=None, update=None, context=None):

    import logging
    import pandas as pd
    import argparse
    import os
    from datetime import datetime
    import sys

    # Налаштування логування
    if telegram_mode:
        # Налаштування логування для телеграм-режиму
        log_level = logging.INFO
        logger = logging.getLogger("TelegramFeatures")
        logger.setLevel(log_level)
    else:
        # Настройка аргументів командного рядка для консольного режиму
        parser = argparse.ArgumentParser(description='Feature Engineering для фінансових часових рядів')
        parser.add_argument('--symbol', type=str, default="BTC-USD", help='Символ для аналізу (наприклад, BTC-USD)')
        parser.add_argument('--start_date', type=str, default="2020-01-01", help='Початкова дата (YYYY-MM-DD)')
        parser.add_argument('--end_date', type=str, default=None, help='Кінцева дата (YYYY-MM-DD)')
        parser.add_argument('--horizon', type=int, default=1, help='Горизонт прогнозування')
        parser.add_argument('--output', type=str, default="features", help='Директорія для збереження результатів')
        parser.add_argument('--log_level', type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            help='Рівень логування')
        parser.add_argument('--feature_groups', type=str, nargs='+',
                            default=None,
                            help='Групи ознак для генерації (розділені пробілами)')

        args = parser.parse_args()

        # Налаштування логування для консольного режиму
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Неправильний рівень логування: {args.log_level}')
        log_level = numeric_level

        # Налаштування логера для консольного режиму
        logger = logging.getLogger("ConsoleFeatures")
        logger.setLevel(log_level)

        # Додаємо обробник для виведення в консоль, якщо його ще немає
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    try:
        # Ініціалізація класу FeatureEngineering
        fe = FeatureEngineering(log_level=log_level)

        if telegram_mode:
            # Отримання параметрів з повідомлення телеграм
            # Це приклад - реальна реалізація буде залежати від структури ваших команд телеграм-бота
            chat_id = update.effective_chat.id
            message_parts = update.message.text.split()

            # Приклад парсингу команди з телеграм
            # Формат: /features symbol start_date [end_date] [horizon]
            if len(message_parts) < 3:
                bot.send_message(chat_id=chat_id,
                                 text="Недостатньо параметрів. Формат: /features symbol start_date [end_date] [horizon]")
                return

            symbol = message_parts[1]
            start_date = message_parts[2]
            end_date = message_parts[3] if len(message_parts) > 3 else None
            horizon = int(message_parts[4]) if len(message_parts) > 4 else 1
            output = "telegram_features"
            feature_groups = None
        else:
            # Використовуємо параметри з командного рядка
            symbol = args.symbol
            start_date = args.start_date
            end_date = args.end_date
            horizon = args.horizon
            output = args.output
            feature_groups = args.feature_groups

        # Перевірка доступності символу
        if symbol not in fe.supported_symbols:
            error_msg = f"Символ {symbol} не підтримується. Доступні символи: {fe.supported_symbols}"
            if telegram_mode:
                bot.send_message(chat_id=chat_id, text=error_msg)
            else:
                logger.error(error_msg)
            return

        # Завантаження даних
        load_msg = f"Завантаження даних для {symbol} з {start_date} до {end_date if end_date else 'сьогодні'}"
        logger.info(load_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=load_msg)

        # Використовуємо db_manager для завантаження даних
        data = fe.db_manager.get_klines_processed(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if data.empty:
            empty_msg = "Отримано порожній набір даних. Перевірте параметри запиту."
            if telegram_mode:
                bot.send_message(chat_id=chat_id, text=empty_msg)
            else:
                logger.error(empty_msg)
            return

        records_msg = f"Завантажено {len(data)} записів"
        logger.info(records_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=records_msg)

        # Створення директорії для виводу, якщо не існує
        if not os.path.exists(output):
            os.makedirs(output)

        # Збереження початкових (сирих) даних
        raw_data_path = os.path.join(output, f"{symbol}_raw_data.csv")
        data.to_csv(raw_data_path, index=True)
        raw_msg = f"Сирі дані збережено до {raw_data_path}"
        logger.info(raw_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=raw_msg)

        # Генерація ознак
        fe_msg = "Початок генерації ознак..."
        logger.info(fe_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=fe_msg)

        # Виконуємо повний конвеєр підготовки ознак
        features_df, target = fe.prepare_features_pipeline(
            data=data,
            target_column='close',
            horizon=horizon,
            feature_groups=feature_groups
        )

        # Додаємо цільову змінну до датафрейму
        features_df[f'target_{horizon}'] = target

        # Збереження датафрейму з ознаками
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        features_path = os.path.join(output, f"{symbol}_features_h{horizon}_{timestamp}.csv")
        features_df.to_csv(features_path, index=True)
        features_msg = f"Датафрейм з {len(features_df.columns) - 1} ознаками збережено до {features_path}"
        logger.info(features_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=features_msg)

        # Створення звіту з описовою статистикою
        stats_report = create_statistics_report(features_df)
        stats_path = os.path.join(output, f"{symbol}_stats_report_{timestamp}.csv")
        stats_report.to_csv(stats_path)
        stats_msg = f"Звіт зі статистикою збережено до {stats_path}"
        logger.info(stats_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=stats_msg)

        # Створення списку ознак з описами
        features_info = create_features_info(features_df)
        info_path = os.path.join(output, f"{symbol}_features_info_{timestamp}.csv")
        features_info.to_csv(info_path, index=False)
        info_msg = f"Інформацію про ознаки збережено до {info_path}"
        logger.info(info_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=info_msg)

        # Якщо консольний режим, виводимо деякі результати у консоль для тестування
        if not telegram_mode:
            print("\n--- Перші 5 рядків датафрейму з ознаками ---")
            print(features_df.head())
            print("\n--- Описова статистика (перші 5 рядків) ---")
            print(stats_report.head())
            print("\n--- Інформація про ознаки (перші 5 рядків) ---")
            print(features_info.head())

        success_msg = "Обробка завершена успішно!"
        logger.info(success_msg)
        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=success_msg)

    except Exception as e:
        error_msg = f"Виникла помилка: {str(e)}"
        logger.error(error_msg)
        import traceback
        trace_msg = traceback.format_exc()
        logger.error(trace_msg)

        if telegram_mode:
            bot.send_message(chat_id=chat_id, text=error_msg)
            # Відправляємо трейсбек помилки, якщо це потрібно
            # bot.send_message(chat_id=chat_id, text=f"```\n{trace_msg}\n```")


def create_statistics_report(data: pd.DataFrame) -> pd.DataFrame:

    # Основна описова статистика
    stats = data.describe().T

    # Додаємо додаткові метрики
    stats['missing'] = data.isnull().sum()
    stats['missing_pct'] = data.isnull().mean() * 100
    stats['unique'] = data.nunique()

    # Для числових стовпців додаємо асиметрію та ексцес
    numeric_cols = data.select_dtypes(include=['number']).columns
    stats.loc[numeric_cols, 'skew'] = data[numeric_cols].skew()
    stats.loc[numeric_cols, 'kurtosis'] = data[numeric_cols].kurtosis()

    return stats


def create_features_info(data: pd.DataFrame) -> pd.DataFrame:

    features = []

    for col in data.columns:
        feature_type = "unknown"
        description = ""

        # Визначаємо тип ознаки на основі префіксу назви
        if col.startswith('lag_') or col.endswith('_lag'):
            feature_type = "lagged"
            description = "Лагова ознака"
        elif col.startswith('rolling_') or '_rolling_' in col:
            feature_type = "rolling"
            description = "Ознака ковзного вікна"
        elif col.startswith('ewm_') or '_ewm_' in col:
            feature_type = "exponential_weighted"
            description = "Експоненційно зважена ознака"
        elif col.startswith('return_') or col.startswith('log_return_'):
            feature_type = "return"
            description = "Ознака прибутковості"
        elif any(col.startswith(x) for x in ['sma_', 'ema_', 'rsi_', 'macd', 'bb_']):
            feature_type = "technical"
            description = "Технічний індикатор"
        elif col.startswith('volatility_') or 'volatility' in col:
            feature_type = "volatility"
            description = "Ознака волатильності"
        elif col.startswith('ratio_'):
            feature_type = "ratio"
            description = "Співвідношення"
        elif col.startswith('target_'):
            feature_type = "target"
            description = "Цільова змінна"
        elif col in ['open', 'high', 'low', 'close', 'volume']:
            feature_type = "price_volume"
            description = "Базовий показник ціни/об'єму"

        # Додаємо більш детальний опис для відомих технічних індикаторів
        if col.startswith('sma_'):
            window = col.split('_')[1]
            description = f"Проста ковзна середня з вікном {window}"
        elif col.startswith('ema_'):
            window = col.split('_')[1]
            description = f"Експоненційна ковзна середня з вікном {window}"
        elif col.startswith('rsi_'):
            window = col.split('_')[1]
            description = f"Індекс відносної сили з вікном {window}"
        elif col.startswith('bb_high_'):
            window = col.split('_')[2]
            description = f"Верхня смуга Боллінджера з вікном {window}"
        elif col.startswith('bb_mid_'):
            window = col.split('_')[2]
            description = f"Середня смуга Боллінджера з вікном {window}"
        elif col.startswith('bb_low_'):
            window = col.split('_')[2]
            description = f"Нижня смуга Боллінджера з вікном {window}"

        # Додаємо інформацію до списку
        features.append({
            'feature_name': col,
            'feature_type': feature_type,
            'description': description,
            'dtype': str(data[col].dtype)
        })

    return pd.DataFrame(features)


def process_and_get_features(self, symbol: str, interval: str = "1d", data_type: str = "klines") -> pd.DataFrame:

    if data_type == "klines":
        df = self.db_manager.get_klines_processed(symbol, interval)
    elif data_type == "orderbook":
        df = self.db_manager.get_orderbook_processed(symbol, interval)
    else:
        raise ValueError("Невідомий тип даних: очікується 'klines' або 'orderbook'")

    # Генерація ознак
    df = self.create_technical_features(df)
    df = self.create_return_features(df)
    df = self.create_volatility_features(df)
    df = self.create_volume_features(df)
    df = self.create_candle_pattern_features(df)
    df = self.create_custom_indicators(df)

    return df


if __name__ == "__main__":
    main(telegram_mode=False)  # За замовчуванням запускаємо в консольному режимі